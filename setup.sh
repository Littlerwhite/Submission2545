#!/bin/bash
# ============================================================
# setup.sh - Reproducibility setup for DeepCFR honeypot paper
# Tested on Ubuntu 22.04 / 24.04
# ============================================================
set -e

OPEN_SPIEL_COMMIT="112b7770"
OPEN_SPIEL_REPO="https://github.com/google-deepmind/open_spiel.git"
REPO_DIR="$HOME/open_spiel"
VENV_DIR="$REPO_DIR/venv"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

log_info "1/7 Deactivating conda (if any)..."
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

log_info "2/7 Installing system dependencies..."
sudo apt update
sudo apt install -y git cmake build-essential python3 python3-pip python3-venv \
    python3-dev libopenblas-dev libeigen3-dev libboost-all-dev \
    virtualenv clang curl wget unzip

log_info "3/7 Cloning OpenSpiel v2.0.1 (commit $OPEN_SPIEL_COMMIT)..."
if [ ! -d "$REPO_DIR" ]; then
    git clone "$OPEN_SPIEL_REPO" "$REPO_DIR"
fi
cd "$REPO_DIR"
git checkout "$OPEN_SPIEL_COMMIT" 2>/dev/null || true

log_info "4/7 Downloading dependencies (may take 5-10 min)..."
timeout 1200 ./install.sh 2>&1 | tail -20 || log_warn "install.sh may have issues"

if [ ! -d "$REPO_DIR/pybind11" ]; then
    log_warn "pybind11 missing, cloning from mirror..."
    git clone --depth 1 https://gitee.com/mirrors/pybind11.git "$REPO_DIR/pybind11" || \
    git clone --depth 1 https://github.com/pybind/pybind11.git "$REPO_DIR/pybind11"
fi

log_info "5/7 Creating virtual environment..."
rm -rf "$VENV_DIR"
env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin \
    /usr/bin/python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

log_info "6/7 Installing Python packages (torch ~200MB)..."
pip install --upgrade pip
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    grep -v "^torch" "$PROJECT_DIR/requirements.txt" > /tmp/req_no_torch.txt
    pip install -r /tmp/req_no_torch.txt
fi
pip install torch --index-url https://download.pytorch.org/whl/cpu

log_info "7/7 Compiling pyspiel (10-25 min, please wait)..."
cd "$REPO_DIR/open_spiel"
rm -rf build
mkdir -p build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python3)
MEM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
MEM_GB=${MEM_GB:-16}
if [ "$MEM_GB" -lt 8 ] 2>/dev/null; then
    make -j2 pyspiel
else
    make -j4 pyspiel
fi

sed -i '/open_spiel\/open_spiel\/build\/python/d' ~/.bashrc 2>/dev/null || true
echo "export PYTHONPATH=\$HOME/open_spiel/open_spiel/build/python:\$HOME/open_spiel:\$PYTHONPATH" >> ~/.bashrc
export PYTHONPATH="$HOME/open_spiel/open_spiel/build/python:$HOME/open_spiel:$PYTHONPATH"

log_info "Copying custom game and algorithm files..."
cp "$PROJECT_DIR/attack_graph_scalable.py" "$REPO_DIR/open_spiel/python/games/"

INIT_FILE="$REPO_DIR/open_spiel/python/games/__init__.py"
sed -i '/attack_graph_scalable/d' "$INIT_FILE" 2>/dev/null || true
echo "from open_spiel.python.games import attack_graph_scalable" >> "$INIT_FILE"

mkdir -p "$REPO_DIR/algorithms"
cp "$PROJECT_DIR/algorithms/"*.py "$REPO_DIR/algorithms/" 2>/dev/null || true
cp "$PROJECT_DIR"/*.py "$REPO_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR"/*.sh "$REPO_DIR/" 2>/dev/null || true

find "$REPO_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo ""
log_info "========== Verification =========="
cd "$REPO_DIR"
python3 -c "import pyspiel; print('pyspiel imported OK')"
python3 -c "
import pyspiel
import open_spiel.python.games.attack_graph_scalable
names = [g.short_name for g in pyspiel.registered_games()]
attack_games = sorted([n for n in names if 'attack_graph' in n])
print('Registered games:', attack_games)
"

echo ""
log_info "========== Setup complete =========="
echo "To run experiments:"
echo "  cd ~/open_spiel"
echo "  source venv/bin/activate"
echo "  python3 run_experiment.py --game attack_graph_10 --algo sdcfr --seed 42 --iterations 50"
