<div align="center">

# Multi-Model RVC Inference
### Simplified RVC Inference for HuggingFace or Google Colab

[![License](https://img.shields.io/github/license/arkandash/Multi-Model-RVC-Inference?style=for-the-badge)](https://github.com/ArkanDash/Multi-Model-RVC-Inference/blob/master/LICENSE)
[![Repository](https://img.shields.io/badge/Github-Multi%20Model%20RVC%20Inference-blue?style=for-the-badge&logo=github)](https://github.com/ArkanDash/Multi-Model-RVC-Inference)
</div>

### Information
Please support the original RVC, without it, this inference wont be possible to make.<br />
[![Original RVC Repository](https://img.shields.io/badge/Github-Original%20RVC%20Repository-blue?style=for-the-badge&logo=github)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
#### Features
- Support V1 & V2 Model ✅
- Youtube Audio Downloader ✅
- Demucs (Voice Splitter) [Internet required for downloading model] ✅
- TTS Support ✅
- Microphone Support ✅
- HuggingFace Spaces Inference [for CPU Tier only] ✅
    - Remove Youtube & Input Path ✅
    - Remove Crepe Support due to gpu requirement ✅

### Automatic Installation
  Install [ffmpeg](https://ffmpeg.org/) first before running these command.
  - Windows
  Run the `start.bat` to download the model and dependencies. <br />
  Run the `run.bat` to run the inference
  - MacOS & Linux
  For MacOS. before running the script, please install [wget](https://formulae.brew.sh/formula/wget) <br />
  Run the `start.sh` to download the model and dependencies. <br />
  Run the `run.sh` to run the inference

### Manual Installation

1. Install Pytorch <br />
    - CPU only (any OS)
    ```bash
    pip install torch torchvision torchaudio
    ```
    - Nvidia (CUDA used)
    ```bash
    # For Windows (Due to flashv2 not supported in windows, Issue: https://github.com/Dao-AILab/flash-attention/issues/345#issuecomment-1747473481)
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    # Other (Linux, etc)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2. Install [ffmpeg](https://ffmpeg.org/)

3. Install Dependencies<br />
```bash
pip install -r requirements.txt
```

4. Download Pre-model 
```bash
# Hubert Model
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt
# Save it to /assets/hubert/hubert_base.pt

# RVMPE (rmvpe pitch extraction, Optional)
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt
# Save it to /assets/rvmpe/rmvpe.pt
```

5. Run WebUI <br />
```bash
python app.py
```

### [How to use](docs/HOW_TO_USE.md)
### [Command Line Arguments](docs/COMMAND_LINE_ARGUMENTS.md)

# Other Inference
[![Advanced RVC Inference](https://img.shields.io/badge/Github-Advanced_RVC_Inference-blue?style=for-the-badge&logo=github)](https://github.com/ArkanDash/Advanced-RVC-Inference)