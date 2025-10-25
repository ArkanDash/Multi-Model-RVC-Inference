@ECHO OFF

rem Check if Python exists
python --version > NUL 2>&1
IF ERRORLEVEL 1 (
  ECHO Python is not installed. Please install Python before running this script.
  EXIT /B 1
)

rem Create virtual environment (.venv)
python -m venv .venv

rem Activate virtual environment
.venv\Scripts\activate

rem Check for Nvidia GPU using nvidia-smi
nvidia-smi > NUL 2>&1
IF ERRORLEVEL 1 (
  rem Install CPU version
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
) ELSE (
  rem Install GPU version
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
)

rem Install dependencies from requirements.txt
pip install -r requirements.txt

rem Download requirement voice model
powershell -Command "Invoke-WebRequest https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt?download=true -OutFile assets/hubert/hubert_base.pt"
powershell -Command "Invoke-WebRequest https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt?download=true -OutFile assets/rvmpe/rmvpe.pt"

rem Run the inference app
python app.py

ECHO Finished!