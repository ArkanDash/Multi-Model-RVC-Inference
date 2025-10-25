import os
import wave
import subprocess
import yt_dlp
import logging

from fairseq import checkpoint_utils
logger = logging.getLogger(__name__)

def load_hubert(config):
    path_check = os.path.exists("assets/hubert/hubert_base.pt")
    if path_check is False:
        logger.warning("hubert_base.pt is missing. Please check the documentation for to get it.")
    else:
        logger.info("hubert_base.pt found.")
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join("assets", "hubert", "hubert_base.pt")],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()
    return hubert_model

def download_audio(url, audio_provider):
    logs = []
    if url == "":
        logs.append("URL required!")
        yield None, "\n".join(logs)
        return None, "\n".join(logs)
    if not os.path.exists("yt"):
        os.mkdir("yt")
    if audio_provider == "Youtube":
        logs.append("Downloading the audio...")
        yield None, "\n".join(logs)
        ydl_opts = {
            'noplaylist': True,
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "outtmpl": 'yt/audio',
        }
        audio_path = "yt/audio.wav"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logs.append("Download Complete.")
        yield audio_path, "\n".join(logs)

def cut_vocal_and_inst(split_model):
    logs = []
    logs.append("Starting the audio splitting process...")
    yield "\n".join(logs), None, None, None
    command = f"demucs --two-stems=vocals -n {split_model} yt/audio.wav -o output"
    result = subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True)
    for line in result.stdout:
        logs.append(line)
        yield "\n".join(logs), None, None, None
    logger.info(result.stdout)
    vocal = f"output/{split_model}/audio/vocals.wav"
    inst = f"output/{split_model}/audio/no_vocals.wav"
    logs.append("Audio splitting complete.")
    yield "\n".join(logs), vocal, inst, vocal

def combine_vocal_and_inst(audio_data, vocal_volume, inst_volume, split_model):
    if not os.path.exists("output/result"):
        os.mkdir("output/result")
    vocal_path = "output/result/output.wav"
    output_path = "output/result/combine.mp3"
    inst_path = f"output/{split_model}/audio/no_vocals.wav"
    with wave.open(vocal_path, "w") as wave_file:
        wave_file.setnchannels(1) 
        wave_file.setsampwidth(2)
        wave_file.setframerate(audio_data[0])
        wave_file.writeframes(audio_data[1].tobytes())
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [0:a]volume={inst_volume}[i];[1:a]volume={vocal_volume}[v];[i][v]amix=inputs=2:duration=longest[a] -map [a] -b:a 320k -c:a libmp3lame {output_path}'
    result = subprocess.run(command.split(), stdout=subprocess.PIPE)
    logger.info(result.stdout.decode())
    return output_path