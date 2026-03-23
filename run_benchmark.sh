#!/usr/bin/env bash
#
# LLMBench Desktop Runner
#
# Pulls latest config, runs benchmarks on local GPU or API models,
# and uploads results to the web server.
#
# Usage:
#   ./run_benchmark.sh                          # Run all enabled models on all datasets
#   ./run_benchmark.sh --models gpt-4o mistral-7b --datasets truthfulqa
#   ./run_benchmark.sh --api-only               # Only run API models
#   ./run_benchmark.sh --local-only             # Only run local models
#
# Setup:
#   1. Clone the repo: git clone <repo-url> && cd LLMBench
#   2. Create .env with your API keys (see .env.example)
#   3. pip install -r requirements.txt
#   4. ./run_benchmark.sh
#

set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present
if [ -f .env ]; then
    echo "Loading .env..."
    set -a
    source .env
    set +a
fi

# Pull latest changes
echo "Pulling latest config..."
git pull --rebase 2>/dev/null || echo "Git pull skipped (not a git repo or no remote)"

# Defaults
UPLOAD_URL="${LLMBENCH_UPLOAD_URL:-https://simonmccallum.org.nz/bench}"
UPLOAD_KEY="${UPLOAD_API_KEY:-${ADMIN_PASSWORD:-}}"
MAX_EXAMPLES="${LLMBENCH_MAX_EXAMPLES:-100}"
METHOD="${LLMBENCH_METHOD:-sequential}"

# Parse arguments
MODELS=()
DATASETS=()
API_ONLY=false
LOCAL_ONLY=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --api-only)
            API_ONLY=true
            shift
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        --max-examples)
            MAX_EXAMPLES="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --upload-url)
            UPLOAD_URL="$2"
            shift 2
            ;;
        --upload-key)
            UPLOAD_KEY="$2"
            shift 2
            ;;
        --no-upload)
            UPLOAD_URL=""
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# If no models specified, auto-detect from config
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Auto-detecting models from config/models.yaml..."
    if $API_ONLY; then
        # Extract API models using Python
        MODELS=($(python3 -c "
import yaml
with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)
for vendor, vcfg in cfg.get('api_models', {}).items():
    if vcfg.get('enabled', True):
        for m in vcfg.get('models', []):
            print(m)
" 2>/dev/null)) || true
    elif $LOCAL_ONLY; then
        MODELS=($(python3 -c "
import yaml
with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)
for key, mcfg in cfg.get('local_models', {}).items():
    if mcfg.get('enabled', True):
        print(key)
" 2>/dev/null)) || true
    else
        # All enabled models
        MODELS=($(python3 -c "
import yaml
with open('config/models.yaml') as f:
    cfg = yaml.safe_load(f)
for key, mcfg in cfg.get('local_models', {}).items():
    if mcfg.get('enabled', True):
        print(key)
for vendor, vcfg in cfg.get('api_models', {}).items():
    if vcfg.get('enabled', True):
        for m in vcfg.get('models', []):
            print(m)
" 2>/dev/null)) || true
    fi
fi

# If no datasets specified, use all available
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=($(python3 -c "
import yaml
with open('config/datasets.yaml') as f:
    cfg = yaml.safe_load(f)
for name in cfg.get('datasets', {}):
    print(name)
" 2>/dev/null)) || DATASETS=("truthfulqa" "arc-easy")
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No models found. Check config/models.yaml or specify --models"
    exit 1
fi

echo ""
echo "========================================"
echo "LLMBench Benchmark Runner"
echo "========================================"
echo "Models:       ${MODELS[*]}"
echo "Datasets:     ${DATASETS[*]}"
echo "Max examples: $MAX_EXAMPLES"
echo "Method:       $METHOD"
echo "Upload URL:   ${UPLOAD_URL:-<disabled>}"
echo "========================================"
echo ""

# Build the command
CMD=(python3 service/runner.py
    --models "${MODELS[@]}"
    --datasets "${DATASETS[@]}"
    --max-examples "$MAX_EXAMPLES"
    --method "$METHOD"
)

if [ -n "$UPLOAD_URL" ]; then
    CMD+=(--upload-url "$UPLOAD_URL")
fi
if [ -n "$UPLOAD_KEY" ]; then
    CMD+=(--upload-key "$UPLOAD_KEY")
fi

# Run it
echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}" "${EXTRA_ARGS[@]}"
