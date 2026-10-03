# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import re
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import generate_env
import pytest
import secret_publication
import secret_setup
from secret_publication import SetupError, publish_private, setup_lock
from setup_templates import CONFIG_TEMPLATE, ENV_TEMPLATE


def key_from(path):
    return next(
        line.split("=", 1)[1]
        for line in path.read_text().splitlines()
        if line.startswith("SERVICE_API_KEY=")
    )


def test_creation_keeps_console_secret_free_and_produces_separate_defaults(
    tmp_path, capsys
):
    assert generate_env.main([], root=tmp_path) == 0
    key = key_from(tmp_path / ".env")
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", key)
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err
    assert (tmp_path / "config.toml").read_text(encoding="utf-8") == CONFIG_TEMPLATE
    assert (tmp_path / ".env").read_text(encoding="utf-8") == ENV_TEMPLATE.format(
        api_key=key
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env", "config.toml"]


@pytest.mark.parametrize("force", [False, True])
def test_explicit_display_only_exposes_the_successfully_published_new_key(
    tmp_path, capsys, force
):
    old = None
    if force:
        secret_setup.generate_setup(tmp_path)
        old = (tmp_path / ".env").read_bytes()
    arguments = ["--show-key"] + (["--force"] if force else [])
    assert generate_env.main(arguments, root=tmp_path) == 0
    key = key_from(tmp_path / ".env")
    captured = capsys.readouterr()
    assert f"SERVICE_API_KEY={key}" in captured.out
    assert key not in captured.err
    if old is not None:
        assert (tmp_path / ".env").read_bytes() != old


@pytest.mark.parametrize("arguments", [[], ["--show-key"]])
def test_existing_secret_is_neither_read_repaired_nor_displayed(
    tmp_path, capsys, monkeypatch, arguments
):
    env = tmp_path / ".env"
    content = b"manually-maintained-secret-data\n"
    env.write_bytes(content)
    os.chmod(env, 0o644)
    info = env.stat()
    # Any read of existing setup data would fail the contract.
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **kw: pytest.fail("existing file read")
    )
    assert generate_env.main(arguments, root=tmp_path) == 1
    captured = capsys.readouterr()
    assert content.decode().strip() not in captured.out + captured.err
    assert env.read_bytes() == content
    assert stat.S_IMODE(env.stat().st_mode) == stat.S_IMODE(info.st_mode)
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / ".env.setup.lock").exists()


def test_rotation_replaces_inode_and_preserves_custom_config_without_display(
    tmp_path, capsys
):
    first = secret_setup.generate_setup(tmp_path)
    env = tmp_path / ".env"
    before = env.stat().st_ino
    config = tmp_path / "config.toml"
    config.write_bytes(b"# operator-specific settings\n")
    assert generate_env.main(["--force"], root=tmp_path) == 0
    key = key_from(env)
    captured = capsys.readouterr()
    assert first.api_key != key
    assert key not in captured.out + captured.err
    assert env.stat().st_ino != before
    assert config.read_bytes() == b"# operator-specific settings\n"
    assert first.api_key not in repr(first)


@pytest.mark.parametrize("name", [".env", "config.toml"])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("kind", ["directory", "hardlink", "symlink", "dangling"])
def test_unsafe_targets_are_refused_without_following_or_mutating_them(
    tmp_path, name, force, kind
):
    target = tmp_path / name
    other = tmp_path / "other"
    other.write_bytes(b"preserve external data")
    if kind == "directory":
        target.mkdir()
    elif kind == "hardlink":
        os.link(other, target)
    else:
        try:
            target.symlink_to(other if kind == "symlink" else tmp_path / "missing")
        except OSError as error:
            pytest.skip(f"Host cannot create symlinks: {error.errno}")
    with pytest.raises(SetupError):
        secret_setup.generate_setup(tmp_path, force=force)
    assert other.read_bytes() == b"preserve external data"
    assert target.lstat()
    assert not list(tmp_path.glob(".env.setup-*.tmp"))
    assert not (tmp_path / ".env.setup.lock").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory/mode contract")
@pytest.mark.parametrize("mode", [0o777, 0o770, 0o702])
def test_writable_shared_setup_directory_is_refused(tmp_path, mode):
    root = tmp_path / "shared"
    root.mkdir(mode=mode)
    root.chmod(mode)
    with pytest.raises(SetupError, match="group/others"):
        secret_setup.generate_setup(root)
    assert not list(root.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner/mode contract")
@pytest.mark.parametrize("umask", [0, 0o022, 0o077, 0o777])
def test_private_permissions_are_in_place_before_the_first_write(
    tmp_path, monkeypatch, umask
):
    original = os.fdopen
    seen = []

    def inspect(fd, *args, **kwargs):
        info = os.fstat(fd)
        seen.append((stat.S_IMODE(info.st_mode), info.st_size))
        assert info.st_uid == os.geteuid()
        assert stat.S_IMODE(info.st_mode) in {0o600, 0o644}
        assert not os.get_inheritable(fd)
        return original(fd, *args, **kwargs)

    monkeypatch.setattr(secret_publication.os, "fdopen", inspect)
    previous = os.umask(umask)
    try:
        secret_setup.generate_setup(tmp_path)
    finally:
        os.umask(previous)
    assert seen == [(0o644, 0), (0o600, 0)]
    assert stat.S_IMODE((tmp_path / ".env").stat().st_mode) == 0o600


def test_exclusive_publication_has_one_winner_under_real_concurrent_writers(tmp_path):
    barrier = Barrier(8)
    target = tmp_path / ".env"
    contents = [f"complete-writer-{n}".encode() * 2048 for n in range(8)]

    def writer(content):
        barrier.wait(timeout=10)
        try:
            publish_private(target, content)
            return content
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        winners = [
            result for result in executor.map(writer, contents) if result is not None
        ]
    assert len(winners) == 1
    assert target.read_bytes() == winners[0]
    assert not list(tmp_path.glob(".env.setup-*.tmp"))


@pytest.mark.parametrize("force", [False, True])
def test_setup_lock_refuses_another_process_without_removing_its_lock(tmp_path, force):
    code = (
        "import sys; from pathlib import Path; from generate_env import main; "
        "raise SystemExit(main(sys.argv[2:], root=Path(sys.argv[1])))"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(generate_env.__file__).parent))
    with setup_lock(tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(tmp_path),
                *(["--force"] if force else []),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 1
        assert "Setup is running" in result.stderr
        assert (tmp_path / ".env.setup.lock").is_dir()
        assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".env.setup.lock").exists()


@pytest.mark.parametrize("phase", ["write", "flush", "fsync", "replace"])
def test_handled_failure_preserves_old_secret_and_cleans_private_staging(
    tmp_path, monkeypatch, capsys, phase
):
    result = secret_setup.generate_setup(tmp_path)
    target = tmp_path / ".env"
    before = target.read_bytes()
    sentinel = "injected-sensitive-error-value"

    def fail(*args, **kwargs):
        raise OSError(sentinel)

    if phase in {"fsync", "replace"}:
        monkeypatch.setattr(secret_publication.os, phase, fail)
    else:
        original = os.fdopen

        class BrokenStream:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()

            def write(self, data):
                if phase == "write":
                    fail()
                return self.stream.write(data)

            def flush(self):
                fail()

        monkeypatch.setattr(
            secret_publication.os,
            "fdopen",
            lambda *a, **kw: BrokenStream(original(*a, **kw)),
        )
    assert generate_env.main(["--force", "--show-key"], root=tmp_path) == 1
    captured = capsys.readouterr()
    assert result.api_key not in captured.out + captured.err
    assert sentinel not in captured.out + captured.err
    assert target.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [".env", "config.toml"]


def test_default_config_failure_does_not_publish_a_key(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fixture failure")

    monkeypatch.setattr(secret_setup, "publish_defaults", fail)
    with pytest.raises(OSError):
        secret_setup.generate_setup(tmp_path)
    assert not (tmp_path / ".env").exists()
    assert not (tmp_path / ".env.setup.lock").exists()


def test_competing_config_publication_preserves_the_winners_settings(
    tmp_path, monkeypatch
):
    original = secret_setup.publish_defaults

    def compete(path, content, **kwargs):
        if path.name == "config.toml":
            path.write_bytes(b"# competing config\n")
        return original(path, content, **kwargs)

    monkeypatch.setattr(secret_setup, "publish_defaults", compete)
    result = secret_setup.generate_setup(tmp_path)
    assert not result.config_created
    assert (tmp_path / "config.toml").read_bytes() == b"# competing config\n"
    assert key_from(tmp_path / ".env") == result.api_key


@pytest.mark.parametrize("replace", [False, True])
def test_final_name_is_never_visible_with_partial_data(tmp_path, monkeypatch, replace):
    target = tmp_path / ".env"
    if replace:
        target.write_bytes(b"old complete value")
    original = os.fsync

    def inspect(fd):
        assert (
            target.read_bytes() == b"old complete value"
            if replace
            else not target.exists()
        )
        return original(fd)

    monkeypatch.setattr(secret_publication.os, "fsync", inspect)
    publish_private(target, b"new complete value" * 4096, replace=replace)
    assert target.read_bytes() == b"new complete value" * 4096


@pytest.mark.skipif(os.name != "nt", reason="Native Windows DACL contract")
def test_windows_dacl_is_owner_only_at_creation_and_after_rotation(
    tmp_path, monkeypatch
):
    original = secret_publication._open_private
    checks = []

    def check(path):
        command = (
            "$a=Get-Acl -LiteralPath $args[0]; "
            "$u=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
            "@{protected=$a.AreAccessRulesProtected;owner=$a.GetOwner([System.Security.Principal.SecurityIdentifier]).Value;"
            "user=$u;rules=@($a.Access | ForEach-Object { @{sid=$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value;"
            "type=$_.AccessControlType.ToString();inherited=$_.IsInherited;rights=[int]$_.FileSystemRights} })} | ConvertTo-Json -Depth 4 -Compress"
        )
        # Pass the fixture path as a PowerShell argument, without shell interpolation.
        script = tmp_path / "acl-probe.ps1"
        script.write_text(command, encoding="utf-8")
        completed = subprocess.run(
            ["pwsh", "-NoProfile", "-File", str(script), str(path)],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        assert not completed.stderr
        acl = json.loads(completed.stdout)
        assert acl["protected"] and acl["owner"] == acl["user"]
        assert len(acl["rules"]) == 1
        assert acl["rules"][0] == {
            "sid": acl["user"],
            "type": "Allow",
            "inherited": False,
            "rights": 2032127,
        }
        checks.append(path.name)

    def inspect(path):
        fd = original(path)
        try:
            assert os.fstat(fd).st_size == 0
            assert not os.get_inheritable(fd)
            check(path)
            return fd
        except BaseException:
            os.close(fd)
            path.unlink(missing_ok=True)
            raise

    monkeypatch.setattr(secret_publication, "_open_private", inspect)
    secret_setup.generate_setup(tmp_path)
    check(tmp_path / ".env")
    secret_setup.generate_setup(tmp_path, force=True)
    check(tmp_path / ".env")
    assert len(checks) == 4


@pytest.mark.skipif(os.name != "nt", reason="Native Windows reparse-point contract")
def test_windows_junction_root_is_refused(tmp_path):
    real = tmp_path / "actual"
    real.mkdir()
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(real)],
        capture_output=True,
        timeout=10,
    )
    if completed.returncode:
        pytest.skip("Junction creation is unavailable")
    try:
        with pytest.raises(SetupError):
            secret_setup.generate_setup(junction)
        assert not list(real.iterdir())
    finally:
        junction.rmdir()  # Only the fixture junction itself, never its target tree.


@pytest.mark.skipif(os.name != "nt", reason="Native Windows fail-closed ACL contract")
@pytest.mark.parametrize("failure", ["unsupported", "query", "crt"])
def test_windows_acl_or_handle_failure_happens_before_data_and_cleans_file(
    tmp_path, monkeypatch, failure
):
    import ctypes
    import msvcrt

    import secret_windows

    path = tmp_path / "private-fixture"
    original = secret_windows._bind
    inspected = []

    def fake_volume(*args):
        inspected.append(path.stat().st_size)
        ctypes.cast(args[5], ctypes.POINTER(ctypes.c_ulong)).contents.value = 0
        ctypes.set_last_error(5)
        return failure != "query"

    def bind(library, name, result, arguments):
        if name == "GetVolumeInformationByHandleW" and failure != "crt":
            return fake_volume
        return original(library, name, result, arguments)

    def fail_crt(*args):
        inspected.append(path.stat().st_size)
        raise OSError("fixture CRT failure")

    monkeypatch.setattr(secret_windows, "_bind", bind)
    if failure == "crt":
        monkeypatch.setattr(msvcrt, "open_osfhandle", fail_crt)
    with pytest.raises(OSError):
        secret_windows.open_private_windows(path)
    assert inspected == [0]
    assert not path.exists()


def test_temporary_name_collision_never_removes_or_overwrites_another_file(
    tmp_path, monkeypatch
):
    collision = tmp_path / ".env.setup-fixture.tmp"
    collision.write_bytes(b"previous writer data")
    monkeypatch.setattr(
        secret_publication.secrets, "token_hex", lambda *args: "fixture"
    )
    with pytest.raises(FileExistsError):
        publish_private(tmp_path / ".env", b"new data")
    assert collision.read_bytes() == b"previous writer data"
    assert not (tmp_path / ".env").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX root symlink contract")
def test_posix_linked_setup_directory_is_refused(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(SetupError):
        secret_setup.generate_setup(link)
    assert not list(actual.iterdir())


def test_start_launcher_generates_once_and_keeps_key_out_of_console(tmp_path):
    import shutil

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(generate_env.__file__).parent
    for name in [
        "generate_env.py",
        "secret_setup.py",
        "secret_publication.py",
        "secret_windows.py",
        "setup_templates.py",
    ]:
        shutil.copyfile(source / name, scripts / name)
    launcher = "start.ps1" if os.name == "nt" else "start.sh"
    shutil.copyfile(source / launcher, scripts / launcher)
    commands = tmp_path / "commands"
    commands.mkdir()
    if os.name == "nt":
        fake_docker = commands / "docker.cmd"
        fake_docker.write_bytes(
            b"@echo off\r\necho Fixture compose launch\r\nexit /b 0\r\n"
        )
        argv = ["pwsh", "-NoProfile", "-File", str(scripts / launcher)]
    else:
        fake_docker = commands / "docker"
        fake_docker.write_bytes(b"#!/bin/sh\nprintf 'Fixture compose launch\\n'\n")
        fake_docker.chmod(0o755)
        argv = ["bash", str(scripts / launcher)]
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"])
    first = subprocess.run(
        argv, input="\n", capture_output=True, text=True, timeout=20, env=env
    )
    assert first.returncode == 0, first.stderr
    key = key_from(tmp_path / ".env")
    assert key not in first.stdout + first.stderr
    assert "private editor" in first.stdout
    assert "Fixture compose launch" in first.stdout
    second = subprocess.run(
        argv, input="\n", capture_output=True, text=True, timeout=20, env=env
    )
    assert second.returncode == 0, second.stderr
    assert "Fixture compose launch" in second.stdout
    assert key not in second.stdout + second.stderr
    assert key_from(tmp_path / ".env") == key
