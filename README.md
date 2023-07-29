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
- HuggingFace Spaces Inference [FREE TIER CPU] ✅
    - Remove Youtube & Input Path ✅
    - Remove Crepe Support due to gpu requirement ✅

### Installation

1. Install Requirement <br />
```bash
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

2. Install [ffmpeg](https://ffmpeg.org/)

3. Download [Hubert Model](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/hubert_base.pt) <br />

4. [OPTIONAL] To use rmvpe pitch extraction, download this [rvmpe.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt)

5. Run WebUI <br />
```bash
python app.py
```
# Other Inference
[Simple RVC Inference](https://github.com/ArkanDash/rvc-simple-inference)