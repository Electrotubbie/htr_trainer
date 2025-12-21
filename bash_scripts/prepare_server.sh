#!/bin/bash
set -e

echo "========================================"
echo " GPU SERVER SETUP (Ubuntu 24.04 + CUDA) "
echo " Python 3.11 + venv + PyTorch CUDA     "
echo "========================================"

# --------- SYSTEM UPDATE ----------
echo "=== Обновление системы ==="
sudo apt update && sudo apt upgrade -y

# --------- BASE PACKAGES ----------
echo "=== Установка базовых пакетов ==="
sudo apt install -y \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    ca-certificates

# --------- PYTHON 3.11 ----------
echo "=== Установка Python 3.11 ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev

echo "Проверка Python 3.11:"
python3.11 --version

# --------- PROJECT STRUCTURE ----------
echo "=== Создание структуры проекта ==="
mkdir -p $HOME/projects

git clone https://github.com/Electrotubbie/htr_trainer.git
PROJECT_DIR=$HOME/projects/htr_trainer

mkdir -p $PROJECT_DIR/logs

cd $PROJECT_DIR

# --------- VENV ----------
echo "=== Создание venv (Python 3.11) ==="
python3.11 -m venv venv

echo "=== Активация venv ==="
source venv/bin/activate

echo "=== Проверка Python в venv ==="
python --version

# --------- PIP ----------
echo "=== Обновление pip ==="
pip install --upgrade pip setuptools wheel

# --------- PYTHON BASE LIBS ----------
echo "=== Установка базовых Python библиотек ==="
pip install -r requirements.txt

# --------- CUDA CHECK ----------
echo "=== Проверка CUDA и GPU ==="
python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
EOF

# --------- NVIDIA CHECK ----------
echo "=== Проверка nvidia-smi ==="
nvidia-smi || echo "nvidia-smi недоступен (проверь драйвер)"

# --------- GOOGLE DRIVE ----------
echo "=== Установка gdown (Google Drive) ==="
pip install gdown

echo "=== DOWNLOADING samples.csv ==="
gdown 1R_ZuM_AalY-PrJvt75vypkdIjChWQ_Bt

echo "=== DOWNLOADING DATASET ==="
gdown 18RIxWgGA-C0IQkn8d87iYS1bUt9LeX-b
rm -rf dataset_slice
unzip -qq dataset_slice.zip -d dataset_slice

echo "=== DOWNLOADING SOURCE MODEL ==="
gdown 1VdSY1nxxzzhgzAvat2H0puljspZV8-WQ
rm -rf Kansallisarkisto_cyrillic-htr-model
unzip -qq Kansallisarkisto_cyrillic-htr-model.zip

# --------- FINISH ----------
echo "========================================"
echo " УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО "
echo "========================================"
echo ""