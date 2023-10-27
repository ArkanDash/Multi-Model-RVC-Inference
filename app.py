import os
import glob
import json
import traceback
import logging
import gradio as gr
import numpy as np
import librosa
import torch
import asyncio
import edge_tts
import yt_dlp
import ffmpeg
import subprocess
import sys
import io
import wave
from datetime import datetime
from fairseq import checkpoint_utils
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from config import Config
config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)
spaces = os.getenv("SYSTEM") == "spaces"
force_support = None
if config.unsupported is False:
    if config.device == "mps" or config.device == "cpu":
        force_support = False
else:
    force_support = True

audio_mode = []
f0method_mode = []
f0method_info = ""

if force_support is False or spaces is True:
    if spaces is True:
        audio_mode = ["Upload audio", "TTS Audio"]
    else:
        audio_mode = ["Input path", "Upload audio", "TTS Audio"]
    f0method_mode = ["pm", "harvest"]
    f0method_info = "PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better). (Default: PM)"
else:
    audio_mode = ["Input path", "Upload audio", "Youtube", "TTS Audio"]
    f0method_mode = ["pm", "harvest", "crepe"]
    f0method_info = "PM is fast, Harvest is good but extremely slow, Rvmpe is alternative to harvest (might be better), and Crepe effect is good but requires GPU (Default: PM)"

if os.path.isfile("rmvpe.pt"):
    f0method_mode.insert(2, "rmvpe")

def create_vc_fn(model_name, tgt_sr, net_g, vc, if_f0, version, file_index):
    def vc_fn(
        vc_audio_mode,
        vc_input, 
        vc_upload,
        tts_text,
        tts_voice,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        try:
            logs = []
            print(f"Converting using {model_name}...")
            logs.append(f"Converting using {model_name}...")
            yield "\n".join(logs), None
            if vc_audio_mode == "Input path" or "Youtube" and vc_input != "":
                audio, sr = librosa.load(vc_input, sr=16000, mono=True)
            elif vc_audio_mode == "Upload audio":
                if vc_upload is None:
                    return "You need to upload an audio", None
                sampling_rate, audio = vc_upload
                duration = audio.shape[0] / sampling_rate
                if duration > 20 and spaces:
                    return "Please upload an audio file that is less than 20 seconds. If you need to generate a longer audio file, please use Colab.", None
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio.transpose(1, 0))
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            elif vc_audio_mode == "TTS Audio":
                if len(tts_text) > 100 and spaces:
                    return "Text is too long", None
                if tts_text is None or tts_voice is None:
                    return "You need to enter text and select a voice", None
                asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
                audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
                vc_input = "tts.mp3"
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                vc_input,
                times,
                f0_up_key,
                f0_method,
                file_index,
                # file_big_npy,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                f0_file=None,
            )
            info = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            print(f"{model_name} | {info}")
            logs.append(f"Successfully Convert {model_name}\n{info}")
            yield "\n".join(logs), (tgt_sr, audio_opt)
        except Exception as err:
            info = traceback.format_exc()
            print(info)
            primt(f"Error when using {model_name}.\n{str(err)}")
            yield info, None
    return vc_fn

def load_model():
    categories = []
    if os.path.isfile("weights/folder_info.json"):
        with open("weights/folder_info.json", "r", encoding="utf-8") as f:
            folder_info = json.load(f)
        for category_name, category_info in folder_info.items():
            if not category_info['enable']:
                continue
            category_title = category_info['title']
            category_folder = category_info['folder_path']
            description = category_info['description']
            models = []
            with open(f"weights/{category_folder}/model_info.json", "r", encoding="utf-8") as f:
                models_info = json.load(f)
            for character_name, info in models_info.items():
                if not info['enable']:
                    continue
                model_title = info['title']
                model_name = info['model_path']
                model_author = info.get("author", None)
                model_cover = f"weights/{category_folder}/{character_name}/{info['cover']}"
                model_index = f"weights/{category_folder}/{character_name}/{info['feature_retrieval_library']}"
                cpt = torch.load(f"weights/{category_folder}/{character_name}/{model_name}", map_location="cpu")
                tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
                if_f0 = cpt.get("f0", 1)
                version = cpt.get("version", "v1")
                if version == "v1":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                    model_version = "V1"
                elif version == "v2":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                    model_version = "V2"
                del net_g.enc_q
                print(net_g.load_state_dict(cpt["weight"], strict=False))
                net_g.eval().to(config.device)
                if config.is_half:
                    net_g = net_g.half()
                else:
                    net_g = net_g.float()
                vc = VC(tgt_sr, config)
                print(f"Model loaded: {character_name} / {info['feature_retrieval_library']} | ({model_version})")
                models.append((character_name, model_title, model_author, model_cover, model_version, create_vc_fn(model_name, tgt_sr, net_g, vc, if_f0, version, model_index)))
            category_count += 1
            categories.append([category_title, description, models])
    elif os.path.exists("weights"):
        models = []
        for w_root, w_dirs, _ in os.walk("weights"):
            model_count = 1
            for sub_dir in w_dirs:
                pth_files = glob.glob(f"weights/{sub_dir}/*.pth")
                index_files = glob.glob(f"weights/{sub_dir}/*.index")
                if pth_files == []:
                    print(f"Model [{model_count}/{len(w_dirs)}]: No Model file detected, skipping...")
                    continue
                cpt = torch.load(pth_files[0])
                tgt_sr = cpt["config"][-1]
                cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
                if_f0 = cpt.get("f0", 1)
                version = cpt.get("version", "v1")
                if version == "v1":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                    model_version = "V1"
                elif version == "v2":
                    if if_f0 == 1:
                        net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                    else:
                        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                    model_version = "V2"
                del net_g.enc_q
                print(net_g.load_state_dict(cpt["weight"], strict=False))
                net_g.eval().to(config.device)
                if config.is_half:
                    net_g = net_g.half()
                else:
                    net_g = net_g.float()
                vc = VC(tgt_sr, config)
                if index_files == []:
                    print("Warning: No Index file detected!")
                    index_info = "None"
                    model_index = ""
                else:
                    index_info = index_files[0]
                    model_index = index_files[0]
                print(f"Model loaded [{model_count}/{len(w_dirs)}]: {index_files[0]} / {index_info} | ({model_version})")
                model_count += 1
                models.append((index_files[0][:-4], index_files[0][:-4], "", "", model_version, create_vc_fn(index_files[0], tgt_sr, net_g, vc, if_f0, version, model_index)))
        categories.append(["Models", "", models])
    else:
        categories = []
    return categories

def download_audio(url, audio_provider):
    logs = []
    if url == "":
        logs.append("URL required!")
        yield None, "\n".join(logs)
        return None, "\n".join(logs)
    if not os.path.exists("dl_audio"):
        os.mkdir("dl_audio")
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
            "outtmpl": 'dl_audio/audio',
        }
        audio_path = "dl_audio/audio.wav"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logs.append("Download Complete.")
        yield audio_path, "\n".join(logs)

def cut_vocal_and_inst(split_model):
    logs = []
    logs.append("Starting the audio splitting process...")
    yield "\n".join(logs), None, None, None
    command = f"demucs --two-stems=vocals -n {split_model} dl_audio/audio.wav -o output"
    result = subprocess.Popen(command.split(), stdout=subprocess.PIPE, text=True)
    for line in result.stdout:
        logs.append(line)
        yield "\n".join(logs), None, None, None
    print(result.stdout)
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
    print(result.stdout.decode())
    return output_path

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

def change_audio_mode(vc_audio_mode):
    if vc_audio_mode == "Input path":
        return (
            # Input & Upload
            gr.Textbox.update(visible=True),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "Upload audio":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=True),
            gr.Audio.update(visible=True),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "Youtube":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            # Splitter
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Slider.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Button.update(visible=True),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )
    elif vc_audio_mode == "TTS Audio":
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Checkbox.update(visible=False),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            # Splitter
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=True),
            gr.Dropdown.update(visible=True)
        )

def use_microphone(microphone):
    if microphone == True:
        return gr.Audio.update(source="microphone")
    else:
        return gr.Audio.update(source="upload")

if __name__ == '__main__':
    load_hubert()
    categories = load_model()
    tts_voice_list = asyncio.new_event_loop().run_until_complete(edge_tts.list_voices())
    voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
    with gr.Blocks() as app:
        gr.Markdown(
            "<div align='center'>\n\n"+
            "# Multi Model RVC Inference\n\n"+
            "[![Repository](https://img.shields.io/badge/Github-Multi%20Model%20RVC%20Inference-blue?style=for-the-badge&logo=github)](https://github.com/ArkanDash/Multi-Model-RVC-Inference)\n\n"+
            "</div>"
        )
        if categories == []:
            gr.Markdown(
                "<div align='center'>\n\n"+
                "## No model found, please add the model into weights folder\n\n"+
                "</div>"
            )
        for (folder_title, description, models) in categories:
            with gr.TabItem(folder_title):
                if description:
                    gr.Markdown(f"### <center> {description}")
                with gr.Tabs():
                    if not models:
                        gr.Markdown("# <center> No Model Loaded.")
                        gr.Markdown("## <center> Please add the model or fix your model path.")
                        continue
                    for (name, title, author, cover, model_version, vc_fn) in models:
                        with gr.TabItem(name):
                            with gr.Row():
                                gr.Markdown(
                                    '<div align="center">'
                                    f'<div>{title}</div>\n'+
                                    f'<div>RVC {model_version} Model</div>\n'+
                                    (f'<div>Model author: {author}</div>' if author else "")+
                                    (f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else "")+
                                    '</div>'
                                )
                            with gr.Row():
                                if spaces is False:
                                    with gr.TabItem("Input"):
                                        with gr.Row():
                                            with gr.Column():
                                                vc_audio_mode = gr.Dropdown(label="Input voice", choices=audio_mode, allow_custom_value=False, value="Upload audio")
                                                # Input
                                                vc_input = gr.Textbox(label="Input audio path", visible=False)
                                                # Upload
                                                vc_microphone_mode = gr.Checkbox(label="Use Microphone", value=False, visible=True, interactive=True)
                                                vc_upload = gr.Audio(label="Upload audio file", source="upload", visible=True, interactive=True)
                                                # Youtube
                                                vc_download_audio = gr.Dropdown(label="Provider", choices=["Youtube"], allow_custom_value=False, visible=False, value="Youtube", info="Select provider (Default: Youtube)")
                                                vc_link = gr.Textbox(label="Youtube URL", visible=False, info="Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A", placeholder="https://www.youtube.com/watch?v=...")
                                                vc_log_yt = gr.Textbox(label="Output Information", visible=False, interactive=False)
                                                vc_download_button = gr.Button("Download Audio", variant="primary", visible=False)
                                                vc_audio_preview = gr.Audio(label="Audio Preview", visible=False)
                                                # TTS
                                                tts_text = gr.Textbox(label="TTS text", info="Text to speech input", visible=False)
                                                tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                                            with gr.Column():
                                                vc_split_model = gr.Dropdown(label="Splitter Model", choices=["hdemucs_mmi", "htdemucs", "htdemucs_ft", "mdx", "mdx_q", "mdx_extra_q"], allow_custom_value=False, visible=False, value="htdemucs", info="Select the splitter model (Default: htdemucs)")
                                                vc_split_log = gr.Textbox(label="Output Information", visible=False, interactive=False)
                                                vc_split = gr.Button("Split Audio", variant="primary", visible=False)
                                                vc_vocal_preview = gr.Audio(label="Vocal Preview", visible=False)
                                                vc_inst_preview = gr.Audio(label="Instrumental Preview", visible=False)
                                    with gr.TabItem("Convert"):
                                        with gr.Row():
                                            with gr.Column():
                                                vc_transform0 = gr.Number(label="Transpose", value=0, info='Type "12" to change from male to female voice. Type "-12" to change female to male voice')
                                                f0method0 = gr.Radio(
                                                    label="Pitch extraction algorithm",
                                                    info=f0method_info,
                                                    choices=f0method_mode,
                                                    value="pm",
                                                    interactive=True
                                                )
                                                index_rate1 = gr.Slider(
                                                    minimum=0,
                                                    maximum=1,
                                                    label="Retrieval feature ratio",
                                                    info="(Default: 0.7)",
                                                    value=0.7,
                                                    interactive=True,
                                                )
                                                filter_radius0 = gr.Slider(
                                                    minimum=0,
                                                    maximum=7,
                                                    label="Apply Median Filtering",
                                                    info="The value represents the filter radius and can reduce breathiness.",
                                                    value=3,
                                                    step=1,
                                                    interactive=True,
                                                )
                                                resample_sr0 = gr.Slider(
                                                    minimum=0,
                                                    maximum=48000,
                                                    label="Resample the output audio",
                                                    info="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling",
                                                    value=0,
                                                    step=1,
                                                    interactive=True,
                                                )
                                                rms_mix_rate0 = gr.Slider(
                                                    minimum=0,
                                                    maximum=1,
                                                    label="Volume Envelope",
                                                    info="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used",
                                                    value=1,
                                                    interactive=True,
                                                )
                                                protect0 = gr.Slider(
                                                    minimum=0,
                                                    maximum=0.5,
                                                    label="Voice Protection",
                                                    info="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
                                                    value=0.5,
                                                    step=0.01,
                                                    interactive=True,
                                                )
                                            with gr.Column():
                                                vc_log = gr.Textbox(label="Output Information", interactive=False)
                                                vc_output = gr.Audio(label="Output Audio", interactive=False)
                                                vc_convert = gr.Button("Convert", variant="primary")
                                                vc_vocal_volume = gr.Slider(
                                                    minimum=0,
                                                    maximum=10,
                                                    label="Vocal volume",
                                                    value=1,
                                                    interactive=True,
                                                    step=1,
                                                    info="Adjust vocal volume (Default: 1}",
                                                    visible=False
                                                )
                                                vc_inst_volume = gr.Slider(
                                                    minimum=0,
                                                    maximum=10,
                                                    label="Instrument volume",
                                                    value=1,
                                                    interactive=True,
                                                    step=1,
                                                    info="Adjust instrument volume (Default: 1}",
                                                    visible=False
                                                )
                                                vc_combined_output = gr.Audio(label="Output Combined Audio", visible=False)
                                                vc_combine =  gr.Button("Combine",variant="primary", visible=False)
                                else:
                                    with gr.Column():
                                        vc_audio_mode = gr.Dropdown(label="Input voice", choices=audio_mode, allow_custom_value=False, value="Upload audio")
                                        # Input
                                        vc_input = gr.Textbox(label="Input audio path", visible=False)
                                        # Upload
                                        vc_microphone_mode = gr.Checkbox(label="Use Microphone", value=False, visible=True, interactive=True)
                                        vc_upload = gr.Audio(label="Upload audio file", source="upload", visible=True, interactive=True)
                                        # Youtube
                                        vc_download_audio = gr.Dropdown(label="Provider", choices=["Youtube"], allow_custom_value=False, visible=False, value="Youtube", info="Select provider (Default: Youtube)")
                                        vc_link = gr.Textbox(label="Youtube URL", visible=False, info="Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A", placeholder="https://www.youtube.com/watch?v=...")
                                        vc_log_yt = gr.Textbox(label="Output Information", visible=False, interactive=False)
                                        vc_download_button = gr.Button("Download Audio", variant="primary", visible=False)
                                        vc_audio_preview = gr.Audio(label="Audio Preview", visible=False)
                                        # Splitter
                                        vc_split_model = gr.Dropdown(label="Splitter Model", choices=["hdemucs_mmi", "htdemucs", "htdemucs_ft", "mdx", "mdx_q", "mdx_extra_q"], allow_custom_value=False, visible=False, value="htdemucs", info="Select the splitter model (Default: htdemucs)")
                                        vc_split_log = gr.Textbox(label="Output Information", visible=False, interactive=False)
                                        vc_split = gr.Button("Split Audio", variant="primary", visible=False)
                                        vc_vocal_preview = gr.Audio(label="Vocal Preview", visible=False)
                                        vc_inst_preview = gr.Audio(label="Instrumental Preview", visible=False)
                                        # TTS
                                        tts_text = gr.Textbox(label="TTS text", info="Text to speech input", visible=False)
                                        tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                                    with gr.Column():
                                        vc_transform0 = gr.Number(label="Transpose", value=0, info='Type "12" to change from male to female voice. Type "-12" to change female to male voice')
                                        f0method0 = gr.Radio(
                                            label="Pitch extraction algorithm",
                                            info=f0method_info,
                                            choices=f0method_mode,
                                            value="pm",
                                            interactive=True
                                        )
                                        index_rate1 = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            label="Retrieval feature ratio",
                                            info="(Default: 0.7)",
                                            value=0.7,
                                            interactive=True,
                                        )
                                        filter_radius0 = gr.Slider(
                                            minimum=0,
                                            maximum=7,
                                            label="Apply Median Filtering",
                                            info="The value represents the filter radius and can reduce breathiness.",
                                            value=3,
                                            step=1,
                                            interactive=True,
                                        )
                                        resample_sr0 = gr.Slider(
                                            minimum=0,
                                            maximum=48000,
                                            label="Resample the output audio",
                                            info="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling",
                                            value=0,
                                            step=1,
                                            interactive=True,
                                        )
                                        rms_mix_rate0 = gr.Slider(
                                            minimum=0,
                                            maximum=1,
                                            label="Volume Envelope",
                                            info="Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used",
                                            value=1,
                                            interactive=True,
                                        )
                                        protect0 = gr.Slider(
                                            minimum=0,
                                            maximum=0.5,
                                            label="Voice Protection",
                                            info="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy",
                                            value=0.5,
                                            step=0.01,
                                            interactive=True,
                                        )
                                    with gr.Column():
                                        vc_log = gr.Textbox(label="Output Information", interactive=False)
                                        vc_output = gr.Audio(label="Output Audio", interactive=False)
                                        vc_convert = gr.Button("Convert", variant="primary")
                                        vc_vocal_volume = gr.Slider(
                                            minimum=0,
                                            maximum=10,
                                            label="Vocal volume",
                                            value=1,
                                            interactive=True,
                                            step=1,
                                            info="Adjust vocal volume (Default: 1}",
                                            visible=False
                                        )
                                        vc_inst_volume = gr.Slider(
                                            minimum=0,
                                            maximum=10,
                                            label="Instrument volume",
                                            value=1,
                                            interactive=True,
                                            step=1,
                                            info="Adjust instrument volume (Default: 1}",
                                            visible=False
                                        )
                                        vc_combined_output = gr.Audio(label="Output Combined Audio", visible=False)
                                        vc_combine =  gr.Button("Combine",variant="primary", visible=False)
                        vc_convert.click(
                            fn=vc_fn, 
                            inputs=[
                                vc_audio_mode,
                                vc_input,
                                vc_upload,
                                tts_text,
                                tts_voice,
                                vc_transform0,
                                f0method0,
                                index_rate1,
                                filter_radius0,
                                resample_sr0,
                                rms_mix_rate0,
                                protect0,
                            ], 
                            outputs=[vc_log ,vc_output]
                        )
                        vc_download_button.click(
                            fn=download_audio, 
                            inputs=[vc_link, vc_download_audio], 
                            outputs=[vc_audio_preview, vc_log_yt]
                        )
                        vc_split.click(
                            fn=cut_vocal_and_inst, 
                            inputs=[vc_split_model], 
                            outputs=[vc_split_log, vc_vocal_preview, vc_inst_preview, vc_input]
                        )
                        vc_combine.click(
                            fn=combine_vocal_and_inst,
                            inputs=[vc_output, vc_vocal_volume, vc_inst_volume, vc_split_model],
                            outputs=[vc_combined_output]
                        )
                        vc_microphone_mode.change(
                            fn=use_microphone,
                            inputs=vc_microphone_mode,
                            outputs=vc_upload
                        )
                        vc_audio_mode.change(
                            fn=change_audio_mode,
                            inputs=[vc_audio_mode],
                            outputs=[
                                vc_input,
                                vc_microphone_mode,
                                vc_upload,
                                vc_download_audio,
                                vc_link,
                                vc_log_yt,
                                vc_download_button,
                                vc_split_model,
                                vc_split_log,
                                vc_split,
                                vc_audio_preview,
                                vc_vocal_preview,
                                vc_inst_preview,
                                vc_vocal_volume,
                                vc_inst_volume,
                                vc_combined_output,
                                vc_combine,
                                tts_text,
                                tts_voice
                            ]
                        )
        app.queue(concurrency_count=1, max_size=20, api_open=config.api).launch(share=config.share)