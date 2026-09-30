#!/usr/bin/env bash
# ==============================================================================
# Alterra — End-to-End Pipeline Execution Script
# Smart Scan Strategy for Electronic Warfare (SIH 2026, DRDO PS 26055)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Virtual environment paths
# ------------------------------------------------------------------------------

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Colors
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
NC="\033[0m"

log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}==>${NC} ${BOLD}$1${NC}"
}

log_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}${BOLD}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

# ------------------------------------------------------------------------------
# 1. Virtual Environment Detection & Setup
# ------------------------------------------------------------------------------

setup_env() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "Virtual environment .venv not found."
        log_info "Create it with:"
        echo "    python -m venv .venv"
        echo "    .venv/bin/python -m pip install -r requirements.txt"
        exit 1
    fi

    if [[ ! -x "$VENV_PYTHON" ]]; then
        log_error "Virtual environment Python not found:"
        echo "    $VENV_PYTHON"
        exit 1
    fi

    log_info "Using virtual environment: $VENV_DIR"

    # Put the venv first on PATH so commands such as `python`, `pip`,
    # `alterra`, and `tensorboard` resolve to the venv.
    export PATH="$VENV_DIR/bin:$PATH"
    export VIRTUAL_ENV="$VENV_DIR"

    # Verify Python really comes from the venv.
    if [[ "$(command -v python)" != "$VENV_PYTHON" ]]; then
        log_error "Python is not resolving to the virtual environment."
        log_error "Expected: $VENV_PYTHON"
        log_error "Found:    $(command -v python)"
        exit 1
    fi

    log_info "Python: $(command -v python)"
    log_info "Version: $(python --version)"

    # Install Alterra in editable mode if it is not importable.
    if ! "$VENV_PYTHON" -c "import alterra" &>/dev/null; then
        log_info "Installing Alterra in editable mode..."
        "$VENV_PYTHON" -m pip install -e .
    fi

    # Verify CLI availability.
    if ! command -v alterra &>/dev/null; then
        log_error "Alterra CLI is not available after installation."
        exit 1
    fi

    log_info "Alterra CLI: $(command -v alterra)"
}

# ------------------------------------------------------------------------------
# 2. Simulation Suite
# ------------------------------------------------------------------------------

run_sim_suite() {
    log_step "1. Verifying Gymnasium Environment (check_env)"
    alterra env check --config configs/default_config.yaml

    log_step "2. Previewing Randomly Sampled Emitter Population"
    alterra emitters preview \
        --config configs/default_config.yaml \
        --episode-length 500

    log_step "3. Previewing Custom Scenario (manual_example.yaml)"
    alterra scenario preview \
        --config configs/default_config.yaml \
        --scenario configs/scenarios/manual_example.yaml

    log_step "4. Running Step Rollout Preview (Random Scan Policy)"
    alterra env preview \
        --config configs/default_config.yaml \
        --steps 20

    log_step "5. Evaluating Rollout Metrics (5 episodes, 50 steps each)"
    alterra env metrics \
        --config configs/default_config.yaml \
        --episodes 5 \
        --steps-per-episode 50

    log_step "6. Generating RF Waterfall Spectrogram Plot"
    alterra env plot \
        --config configs/default_config.yaml \
        --steps 150 \
        --out episode_waterfall.png

    log_success "Waterfall spectrogram saved to: episode_waterfall.png"

    log_step "7. Previewing & Exporting Pulse Descriptor Words (PDWs)"
    alterra pdw preview \
        --config configs/default_config.yaml \
        --limit 5

    alterra pdw export \
        --config configs/default_config.yaml \
        --episode-length 500 \
        --out pdws.jsonl

    log_success "PDW stream exported to: pdws.jsonl"
}

# ------------------------------------------------------------------------------
# 3. Dataset Generation
# ------------------------------------------------------------------------------

run_dataset_gen() {
    local episodes="${1:-2}"
    local out_dir="${2:-data/generated}"

    log_step "Generating Batch Dataset ($episodes episodes -> $out_dir)"

    "$VENV_PYTHON" scripts/generate_dataset.py \
        --config configs/default_config.yaml \
        --episodes "$episodes" \
        --out-dir "$out_dir"

    log_success "Dataset generation complete ($out_dir/manifest.json created)."
}

# ------------------------------------------------------------------------------
# 4. RL Smoke Test
# ------------------------------------------------------------------------------

run_rl_smoke_test() {
    log_step "8. RL Scheduler: Quick PPO Training Smoke Test (1,000 steps)"

    mkdir -p model/agents/checkpoints

    "$VENV_PYTHON" -m model.agents.train_ppo \
        --config configs/default_config.yaml \
        --timesteps 1000 \
        --n-envs 2 \
        --checkpoint-freq 5000 \
        --eval-freq 5000 \
        --out model/agents/checkpoints/demo_ppo.zip \
        --tensorboard-log model/agents/tb_logs

    log_step "9. RL Scheduler: Evaluating Demo PPO Checkpoint"

    "$VENV_PYTHON" -m model.agents.evaluate \
        --config configs/default_config.yaml \
        --model model/agents/checkpoints/demo_ppo.zip \
        --episodes 3 \
        --steps-per-episode 50

    log_step "10. RL Scheduler: Inspecting Action Diversity"

    "$VENV_PYTHON" -m model.agents.inspect_policy \
        --config configs/default_config.yaml \
        --model model/agents/checkpoints/demo_ppo.zip \
        --steps 50 \
        --episodes 2
}

# ------------------------------------------------------------------------------
# 5. Full RL Training
# ------------------------------------------------------------------------------

run_full_rl_train() {
    local timesteps="${1:-500000}"
    local n_envs="${2:-4}"
    local out_file="${3:-model/agents/checkpoints/ppo_scheduler_full.zip}"

    log_step "Full PPO Training Run: $timesteps timesteps on $n_envs parallel envs"

    "$VENV_PYTHON" -m model.agents.train_ppo \
        --config configs/default_config.yaml \
        --timesteps "$timesteps" \
        --n-envs "$n_envs" \
        --ent-coef 0.02 \
        --out "$out_file" \
        --tensorboard-log model/agents/tb_logs

    log_success "PPO model saved to: $out_file"
}

# ------------------------------------------------------------------------------
# 6. Usage
# ------------------------------------------------------------------------------

usage() {
    cat <<EOF
Alterra — Smart Scan Strategy Execution Script

USAGE:
    ./run.sh [OPTIONS]

OPTIONS:
    --all, -a            Run the complete end-to-end demonstration (default)
                         (Simulation, Metrics, Waterfall Plot, PDW, Dataset sample, RL Smoke Test)

    --sim, -s            Run only the RF simulation, environment, and CLI tools

    --dataset, -d [N]    Generate N dataset episodes (default: 5)
                         to data/generated

    --train-rl [STEPS]   Train the PPO RL scheduler
                         (default: 500000 timesteps)

    --smoke-rl           Run a quick 1000-step RL smoke test & evaluation

    --eval-rl <MODEL>    Evaluate a trained PPO model checkpoint

    --tb, --dashboard    Launch TensorBoard dashboard on http://localhost:6006

    --gui, -g            Launch the C++ Qt6 GUI desktop application

    --help, -h           Show this help message

EXAMPLES:
    ./run.sh
    ./run.sh --sim
    ./run.sh --train-rl 200000
    ./run.sh --eval-rl model/agents/checkpoints/demo_ppo.zip
    ./run.sh --tb
    ./run.sh --gui
EOF
}

# ------------------------------------------------------------------------------
# 7. Main Dispatcher
# ------------------------------------------------------------------------------

main() {
    setup_env

    local mode="${1:---all}"

    case "$mode" in

        --all|-a)
            echo -e "${BOLD}${GREEN}====================================================${NC}"
            echo -e "${BOLD}${GREEN}   Running Alterra Full End-to-End Demonstration    ${NC}"
            echo -e "${BOLD}${GREEN}====================================================${NC}"

            run_sim_suite
            run_dataset_gen 2 "data/demo_dataset"
            run_rl_smoke_test

            echo -e "\n${BOLD}${GREEN}====================================================${NC}"
            echo -e "${BOLD}${GREEN}   All Alterra Pipeline Components Passed Successfully!${NC}"
            echo -e "${BOLD}${GREEN}====================================================${NC}"
            ;;

        --sim|-s)
            run_sim_suite
            ;;

        --dataset|-d)
            local count="${2:-5}"
            run_dataset_gen "$count" "data/generated"
            ;;

        --smoke-rl)
            run_rl_smoke_test
            ;;

        --train-rl)
            local steps="${2:-500000}"
            local envs="${3:-4}"

            run_full_rl_train "$steps" "$envs"
            ;;

        --eval-rl)
            if [[ -z "${2:-}" ]]; then
                log_error "Please provide a model path:"
                echo "    ./run.sh --eval-rl <path_to_model.zip>"
                exit 1
            fi

            "$VENV_PYTHON" -m model.agents.evaluate \
                --config configs/default_config.yaml \
                --model "$2" \
                --episodes 10 \
                --steps-per-episode 100
            ;;

        --tb|--dashboard)
            local port="${2:-6006}"

            log_step "Launching TensorBoard on http://localhost:$port (Press CTRL+C to quit)..."

            "$VENV_PYTHON" -m tensorboard.main \
                --logdir model/agents/tb_logs \
                --port "$port"
            ;;

        --gui|-g)
            log_step "Launching Alterra C++ Qt GUI..."

            if [[ ! -f "interfaces/qt_gui/build/alterra_gui" ]]; then
                log_info "Building alterra_gui binary with cmake..."

                local cmake_prefix=""
                local pybind_dir

                pybind_dir="$("$VENV_PYTHON" -m pybind11 --cmakedir 2>/dev/null || true)"

                if [[ "$OSTYPE" == "darwin"* ]]; then
                    local qt_prefix

                    qt_prefix="$(
                        brew --prefix qt@6 2>/dev/null ||
                        brew --prefix qt 2>/dev/null ||
                        brew --prefix qtbase 2>/dev/null ||
                        echo '/opt/homebrew/opt/qtbase'
                    )"

                    cmake_prefix="${qt_prefix};${pybind_dir}"
                else
                    cmake_prefix="${pybind_dir}"
                fi

                cmake -B interfaces/qt_gui/build \
                    -S interfaces/qt_gui \
                    -DCMAKE_PREFIX_PATH="$cmake_prefix" \
                    -DPython_EXECUTABLE="$VENV_PYTHON"

                cmake --build interfaces/qt_gui/build
            fi

            ./interfaces/qt_gui/build/alterra_gui \
                "${2:-configs/default_config.yaml}" \
                "${3:-model/agents/checkpoints/best/best_model.zip}"
            ;;

        --help|-h)
            usage
            ;;

        *)
            log_error "Unknown option: $mode"
            usage
            exit 1
            ;;
    esac
}

main "$@"
