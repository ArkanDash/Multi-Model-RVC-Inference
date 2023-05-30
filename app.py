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
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from vc_infer_pipeline import VC
from config import Config
config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index):
    def vc_fn(
        input_audio,
        upload_audio,
        upload_mode,
        f0_up_key,
        f0_method,
        index_rate,
        tts_mode,
        tts_text,
        tts_voice
    ):
        try:
            if tts_mode:
                if tts_text is None or tts_voice is None:
                    return "You need to enter text and select a voice", None
                asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save("tts.mp3"))
                audio, sr = librosa.load("tts.mp3", sr=16000, mono=True)
            else:
                if upload_mode:
                    if input_audio is None:
                        return "You need to upload an audio", None
                    sampling_rate, audio = upload_audio
                    duration = audio.shape[0] / sampling_rate
                    audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio.transpose(1, 0))
                    if sampling_rate != 16000:
                        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            times = [0, 0, 0]
            f0_up_key = int(f0_up_key)
            audio_opt = vc.pipeline(
                hubert_model,
                net_g,
                0,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                if_f0,
                f0_file=None,
            )
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s"
            )
            return (tgt_sr, audio_opt)
        except:
            info = traceback.format_exc()
            print(info)
            return info, (None, None)
    return vc_fn

def cut_vocal_and_inst(url, audio_provider, split_model):
    if url != "":
        if not os.path.exists("dl_audio"):
            os.mkdir("dl_audio")
        if audio_provider == "Youtube":
            ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "outtmpl": 'dl_audio/youtube_audio',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            audio_path = "dl_audio/youtube_audio.wav"
        else:
            # Spotify doesnt work.
            # Need to find other solution soon.
            ''' 
            command = f"spotdl download {url} --output dl_audio/.wav"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            audio_path = "dl_audio/spotify_audio.wav"
            '''
        if split_model == "htdemucs":
            command = f"demucs --two-stems=vocals {audio_path} -o output"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            return "output/htdemucs/youtube_audio/vocals.wav", "output/htdemucs/youtube_audio/no_vocals.wav", audio_path, "output/htdemucs/youtube_audio/vocals.wav"
        else:
            command = f"demucs --two-stems=vocals -n mdx_extra_q {audio_path} -o output"
            result = subprocess.run(command.split(), stdout=subprocess.PIPE)
            print(result.stdout.decode())
            return "output/mdx_extra_q/youtube_audio/vocals.wav", "output/mdx_extra_q/youtube_audio/no_vocals.wav", audio_path, "output/mdx_extra_q/youtube_audio/vocals.wav"
    else:
        raise gr.Error("URL Required!")
        return None, None, None, None

def combine_vocal_and_inst(audio_data, audio_volume, split_model):
    if not os.path.exists("output/result"):
        os.mkdir("output/result")
    vocal_path = "output/result/output.wav"
    output_path = "output/result/combine.mp3"
    if split_model == "htdemucs":
        inst_path = "output/htdemucs/youtube_audio/no_vocals.wav"
    else:
        inst_path = "output/mdx_extra_q/youtube_audio/no_vocals.wav"
    with wave.open(vocal_path, "w") as wave_file:
        wave_file.setnchannels(1) 
        wave_file.setsampwidth(2)
        wave_file.setframerate(audio_data[0])
        wave_file.writeframes(audio_data[1].tobytes())
    command =  f'ffmpeg -y -i {inst_path} -i {vocal_path} -filter_complex [1:a]volume={audio_volume}dB[v];[0:a][v]amix=inputs=2:duration=longest -b:a 320k -c:a libmp3lame {output_path}'
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

def change_to_tts_mode(tts_mode, upload_mode):
    if tts_mode:
        return gr.Textbox.update(visible=False), gr.Audio.update(visible=False), gr.Checkbox.update(visible=False), gr.Textbox.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        if upload_mode:
            return gr.Textbox.update(visible=False), gr.Audio.update(visible=True), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)
        else:
            return gr.Textbox.update(visible=True), gr.Audio.update(visible=False), gr.Checkbox.update(visible=True), gr.Textbox.update(visible=False), gr.Dropdown.update(visible=False)

def change_to_upload_mode(upload_mode):
    if upload_mode:
        return gr.Textbox().update(visible=False), gr.Audio().update(visible=True)
    else:
        return gr.Textbox().update(visible=True), gr.Audio().update(visible=False)

if __name__ == '__main__':
    load_hubert()
    categories = []
    tts_voice_list = asyncio.get_event_loop().run_until_complete(edge_tts.list_voices())
    voices = [f"{v['ShortName']}-{v['Gender']}" for v in tts_voice_list]
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
        for model_name, info in models_info.items():
            if not info['enable']:
                continue
            model_title = info['title']
            model_author = info.get("author", None)
            model_cover = f"weights/{category_folder}/{model_name}/{info['cover']}"
            model_index = f"weights/{category_folder}/{model_name}/{info['feature_retrieval_library']}"
            cpt = torch.load(f"weights/{category_folder}/{model_name}/{model_name}.pth", map_location="cpu")
            tgt_sr = cpt["config"][-1]
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
            if_f0 = cpt.get("f0", 1)
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            del net_g.enc_q
            print(net_g.load_state_dict(cpt["weight"], strict=False))
            net_g.eval().to(config.device)
            if config.is_half:
                net_g = net_g.half()
            else:
                net_g = net_g.float()
            vc = VC(tgt_sr, config)
            print(f"Model loaded: {model_name}")
            models.append((model_name, model_title, model_author, model_cover, create_vc_fn(tgt_sr, net_g, vc, if_f0, model_index)))
        categories.append([category_title, category_folder, description, models])
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> RVC Models\n"
            "## <center> The input audio should be clean and pure voice without background music.\n"
            "### <center> This project was inspired by [zomehwh](https://huggingface.co/spaces/zomehwh/rvc-models) and [ardha27](https://huggingface.co/spaces/ardha27/rvc-models)\n"
            "[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/110kiMZTdP6Ri1lY9-NbQf17GVPPhHyeT?usp=sharing)\n\n"
            "[![Original Repo](https://badgen.net/badge/icon/github?icon=github&label=Original%20Repo)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)"
        )
        for (folder_title, folder, description, models) in categories:
            with gr.TabItem(folder_title):
                if description:
                    gr.Markdown(f"### <center> {description}")
                with gr.Tabs():
                    if not models:
                        gr.Markdown("# <center> No Model Loaded.")
                        gr.Markdown("## <center> Please added the model or fix your model path.")
                        continue
                    for (name, title, author, cover, vc_fn) in models:
                        with gr.TabItem(name):
                            with gr.Row():
                                gr.Markdown(
                                    '<div align="center">'
                                    f'<div>{title}</div>\n'+
                                    (f'<div>Model author: {author}</div>' if author else "")+
                                    (f'<img style="width:auto;height:300px;" src="file/{cover}">' if cover else "")+
                                    '</div>'
                                )
                            with gr.Row():
                                with gr.Column():
                                    vc_download_audio = gr.Dropdown(label="Provider", choices=["Youtube"], allow_custom_value=False, value="Youtube", info="Select provider [REQUIRED: UPLOAD MODE = OFF] (Default: Youtube)")
                                    vc_link = gr.Textbox(label="Youtube URL", info="Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A")
                                    vc_split_model = gr.Dropdown(label="Splitter Model", choices=["htdemucs", "mdx_extra_q"], allow_custom_value=False, value="htdemucs", info="Select the splitter model (Default: htdemucs)")
                                    vc_split = gr.Button("Split Audio", variant="primary")
                                    vc_vocal_preview = gr.Audio(label="Vocal Preview")
                                    vc_inst_preview = gr.Audio(label="Instrumental Preview")
                                    vc_audio_preview = gr.Audio(label="Audio Preview")
                                with gr.Column():
                                    upload_mode = gr.Checkbox(label="Upload mode", value=False, info="Enable to upload audio instead of audio path")
                                    vc_input = gr.Textbox(label="Input audio path")
                                    vc_upload = gr.Audio(label="Upload audio file", visible=False, interactive=True)
                                    vc_transpose = gr.Number(label="Transpose", value=0, info='Type "12" to change from male to female voice. Type "-12" to change female to male voice')
                                    vc_f0method = gr.Radio(
                                        label="Pitch extraction algorithm",
                                        choices=["pm", "harvest"],
                                        value="pm",
                                        interactive=True,
                                        info="PM is fast but Harvest is better for low frequencies. (Default: PM)"
                                    )
                                    vc_index_ratio = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label="Retrieval feature ratio",
                                        value=0.6,
                                        interactive=True,
                                        info="(Default: 0.6)"
                                    )
                                    tts_mode = gr.Checkbox(label="tts (use edge-tts as input)", value=False)
                                    tts_text = gr.Textbox(visible=False, label="TTS text")
                                    tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                                    vc_output = gr.Audio(label="Output Audio", interactive=False)
                                    vc_submit = gr.Button("Convert", variant="primary")
                                with gr.Column():
                                    vc_volume = gr.Slider(
                                        minimum=0,
                                        maximum=10,
                                        label="Vocal volume",
                                        value=4,
                                        interactive=True,
                                        step=1,
                                        info="Adjust vocal volume (Default: 4}"
                                    )
                                    vc_combined_output = gr.Audio(label="Output Combined Audio")
                                    vc_combine =  gr.Button("Combine",variant="primary")
                        vc_submit.click(vc_fn, [vc_input, vc_upload, upload_mode, vc_transpose, vc_f0method, vc_index_ratio, tts_mode, tts_text, tts_voice], [vc_output])
                        vc_split.click(cut_vocal_and_inst, [vc_link, vc_download_audio, vc_split_model], [vc_vocal_preview, vc_inst_preview, vc_audio_preview, vc_input])
                        vc_combine.click(combine_vocal_and_inst, [vc_output, vc_volume, vc_split_model], vc_combined_output)
                        tts_mode.change(change_to_tts_mode, [tts_mode, upload_mode], [vc_input, vc_upload, upload_mode, tts_text, tts_voice])
                        upload_mode.change(change_to_upload_mode, [upload_mode], [vc_input, vc_upload])
        app.queue(concurrency_count=1, max_size=20, api_open=config.api).launch(share=config.colab)