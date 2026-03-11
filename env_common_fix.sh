#!/usr/bin/env bash

COMMON_DIR=$(uv run python -c 'import pathlib,ylabcommon;print(pathlib.Path(ylabcommon.__file__).parent)')

source "$COMMON_DIR/diagnostics/fix_env.sh"
