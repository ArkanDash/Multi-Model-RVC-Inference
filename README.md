<div align="center">

# Multi Model RVC Inference

</div>

### Information
This Inference Pipeline is from Mangio-Fork <br />
Support V1 Model and V2 Model

Original Repository: [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) <br />
Mangio-Fork Repository: [RVC-Mangio-Fork-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

#### Features
- Youtube Audio Downloader
- Demucs (Voice Splitter) [Internet Required] 
- TTS Support
- HuggingFace Spaces Inference [FREE TIER CPU]
    - Remove Youtube & Input Path
    - Remove Crepe Support due to gpu requirement

#### Plans
- New Multi RVC Inference ❗
- Model Downloader

### Installation
[Download Hubert Model](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

Install Requirement
```bash
pip install -r requirements.txt
```

Run WebUI
```bash
python app.py
```

Share gradio
```bash
python app.py --colab
```

<div align="center">

[![GitHub](https://img.shields.io/github/license/arkandash/Multi-Model-RVC-Inference)](https://github.com/ArkanDash/Multi-Model-RVC-Inference/blob/master/LICENSE)
</div>