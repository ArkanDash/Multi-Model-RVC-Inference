#!/bin/bash

# Check if Python exists
if ! command -v python &> /dev/null; then
  echo "Python is not installed. Please install Python before running this script."
  exit 1
fi

# Create virtual environment (.venv)
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Check for Nvidia GPU using nvidia-smi
if nvidia-smi &> /dev/null; then
  # Install GPU version
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
else
  # Install CPU version
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download requirement voice model
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true -O assets/hubert/hubert_base.pt
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true -O assets/rvmpe/rmvpe.pt


# Run the inference app
python app.py

echo "Finished!"
