"""Tests for __main__ CLI module."""

import sys
import os
from unittest.mock import patch
import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


def test_main_no_args_returns_error(tmp_path):
    old_cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(tmp_path)
        sys.argv = ["rag"]
        from codebase_rag.dev.__main__ import main
        with patch("sys.stderr", new_callable=lambda: __import__("io").StringIO) as mock_err:
            ret = main()
        assert ret == 1
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
