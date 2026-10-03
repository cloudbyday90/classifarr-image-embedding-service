# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import sys

import capacity_client_process as owner
import pytest


def child(monkeypatch, tmp_path, code):
    script = tmp_path / "child.py"
    script.write_text(code, encoding="utf-8")
    monkeypatch.setattr(
        owner, "client_command", lambda: [sys.executable, "-I", str(script)]
    )
    processes = []
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(owner.asyncio, "create_subprocess_exec", spawn)
    return processes


def test_child_receives_only_explicit_ipc_without_ambient_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SERVICE_API_KEY", "ambient-private-value")
    monkeypatch.setenv("PYTHONPATH", "untrusted-loader")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    processes = child(
        monkeypatch,
        tmp_path,
        "import sys,json,os\nmessage=json.load(sys.stdin)\nassert message['key']=='private-ipc-value'\nassert not any(k in os.environ for k in ['SERVICE_API_KEY','PYTHONPATH','HTTPS_PROXY'])\nprint(json.dumps({'passed':True}))\n",
    )
    result = asyncio.run(owner.run_client({"key": "private-ipc-value"}))
    assert result == {"passed": True, "reaped": True, "returncode": 0}
    assert all(process.returncode == 0 for process in processes)
    assert "private-ipc-value" not in str(result)


@pytest.mark.parametrize("reason", ["timeout", "overflow", "nonzero", "invalid"])
def test_child_failure_always_reaps_and_bounds_output(monkeypatch, tmp_path, reason):
    code = {
        "timeout": "import sys,time\nsys.stdin.read()\ntime.sleep(30)\n",
        "overflow": "import sys,time\nsys.stdin.read()\nsys.stdout.write('x'*2000000)\nsys.stdout.flush()\ntime.sleep(30)\n",
        "nonzero": "import sys\nsys.stdin.read()\nprint('private-response')\nsys.exit(1)\n",
        "invalid": "import sys\nsys.stdin.read()\nprint('{}')\n",
    }[reason]
    processes = child(monkeypatch, tmp_path, code)
    monkeypatch.setattr(owner, "MAX_OUTPUT_BYTES", 1024)
    with pytest.raises(Exception) as error:
        asyncio.run(owner.run_client({}, timeout=0.3 if reason == "timeout" else 5))
    assert "private-response" not in str(error.value)
    assert len(processes) == 1 and processes[0].returncode is not None


def test_cancellation_kills_and_reaps_child(monkeypatch, tmp_path):
    processes = child(
        monkeypatch, tmp_path, "import sys,time\nsys.stdin.read()\ntime.sleep(30)\n"
    )

    async def scenario():
        task = asyncio.create_task(owner.run_client({}))
        async with asyncio.timeout(5):
            while not processes:
                await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None

    asyncio.run(scenario())


def test_cancellation_during_launch_still_reaps_child(monkeypatch, tmp_path):
    processes = child(monkeypatch, tmp_path, "import time\ntime.sleep(30)\n")
    original = owner.asyncio.create_subprocess_exec

    async def scenario():
        release = asyncio.Event()

        async def slow_spawn(*args, **kwargs):
            process = await original(*args, **kwargs)
            await release.wait()
            return process

        monkeypatch.setattr(owner.asyncio, "create_subprocess_exec", slow_spawn)
        task = asyncio.create_task(owner.run_client({}))
        async with asyncio.timeout(5):
            while not processes:
                await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None

    asyncio.run(scenario())


def test_oversized_ipc_refused_without_starting_interpreter(monkeypatch):
    monkeypatch.setattr(
        owner, "client_command", lambda: pytest.fail("Unexpected child")
    )
    with pytest.raises(ValueError, match="input exceeds"):
        asyncio.run(owner.run_client({"key": "x" * owner.MAX_INPUT_BYTES}))
