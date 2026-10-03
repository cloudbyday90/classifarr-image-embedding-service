# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fresh-interpreter entrypoint; only the parent may select this worker."""

import sys

from .remote_fetch import fetch_remote_image
from .remote_protocol import MAX_REQUEST_BYTES, decode_request, encode_error


def main() -> None:
    try:
        url, options = decode_request(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1))
        data = fetch_remote_image(url, options)
        result = b"O" + data
    except Exception as error:
        result = encode_error(error)
    sys.stdout.buffer.write(result)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
