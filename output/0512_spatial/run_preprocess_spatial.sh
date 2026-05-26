#!/usr/bin/env bash
#
# run_preprocess_spatial.sh
#
# Wrapper script to run preprocess_spatial.py with a YAML config file.
#
# Usage:
#   ./run_preprocess_spatial.sh                          # uses default config
#   ./run_preprocess_spatial.sh path/to/config.yaml      # custom config
#   ./run_preprocess_spatial.sh --raw_h5ad /path/to/input.h5ad  # override input
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="/root/code/nicheformer/output/0512_spatial"
PROJECT_DIR="/root/code/nicheformer"
VENV_DIR="$PROJECT_DIR/.venv-cxg"

# config file lives next to this script
# CONFIG="$SCRIPT_DIR/preprocess_spatial_config.yaml"

CONFIG="/root/code/nicheformer/output/0518/preprocess_perturb_map_config.yaml"

# Use first argument as config path if it ends in .yaml or .yml,
# otherwise use the default config.
if [ $# -ge 1 ] && [[ "$1" =~ \.yaml$|\.yml$ ]]; then
    CONFIG_PATH="$1"
    shift
else
    CONFIG_PATH="$CONFIG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[run_preprocess_spatial] ERROR: Config file not found: $CONFIG_PATH" >&2
    exit 1
fi

# ============================================================================
# Locate the Python interpreter
# ============================================================================
if [ -d "$VENV_DIR" ]; then
    PYTHON="$VENV_DIR/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "[run_preprocess_spatial] Using Python: $PYTHON"
echo "[run_preprocess_spatial] Project dir: $PROJECT_DIR"
echo "[run_preprocess_spatial] Config: $CONFIG_PATH"
echo "[run_preprocess_spatial] Extra args: $*"

# ============================================================================
# Ensure the nicheformer package is importable
# ============================================================================
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PROJECT_DIR/src"

# ============================================================================
# Run the preprocessing script
# ============================================================================
cd "$PROJECT_DIR"

exec "$PYTHON" "$SCRIPT_DIR/preprocess_spatial.py" --config-yaml "$CONFIG_PATH" "$@"
