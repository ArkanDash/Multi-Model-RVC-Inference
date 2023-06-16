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
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from config import Config
config = Config()
logging.getLogger("numba").setLevel(logging.WARNING)

def create_vc_fn(tgt_sr, net_g, vc, if_f0, file_index):
    def vc_fn(
        vc_audio_mode,
        vc_input, 
        vc_upload,
        tts_text,
        tts_voice,
        spk_item,
        f0_up_key,
        f0_method,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        try:
            if vc_audio_mode == "Input path" or "Youtube" and vc_input != "":
                audio, sr = librosa.load(vc_input, sr=16000, mono=True)
            elif vc_audio_mode == "Upload audio":
                if vc_upload is None:
                    return "You need to upload an audio", None
                sampling_rate, audio = vc_upload
                duration = audio.shape[0] / sampling_rate
                audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio.transpose(1, 0))
                if sampling_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            elif vc_audio_mode == "TTS Audio":
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
                spk_item,
                audio,
                vc_input,
                times,
                f0_up_key,
                f0_method,
                file_index,
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
            print(info)
            return info, (tgt_sr, audio_opt)
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

def change_audio_mode(vc_audio_mode):
    if vc_audio_mode == "Input path":
        return (
            # Input & Upload
            gr.Textbox.update(visible=True),
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
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
            gr.Audio.update(visible=True),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
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
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Dropdown.update(visible=True),
            gr.Button.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
            gr.Audio.update(visible=True),
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
            gr.Audio.update(visible=False),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=True),
            gr.Dropdown.update(visible=True)
        )
    else:
        return (
            # Input & Upload
            gr.Textbox.update(visible=False),
            gr.Audio.update(visible=True),
            # Youtube
            gr.Dropdown.update(visible=False),
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False),
            gr.Button.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Slider.update(visible=False),
            gr.Audio.update(visible=False),
            gr.Button.update(visible=False),
            # TTS
            gr.Textbox.update(visible=False),
            gr.Dropdown.update(visible=False)
        )

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
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
                nodel_version = "V1"
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
                nodel_version = "V2"
            del net_g.enc_q
            print(net_g.load_state_dict(cpt["weight"], strict=False))
            net_g.eval().to(config.device)
            if config.is_half:
                net_g = net_g.half()
            else:
                net_g = net_g.float()
            vc = VC(tgt_sr, config)
            print(f"Model loaded: {model_name}")
            models.append((model_name, model_title, model_author, model_cover, nodel_version, create_vc_fn(tgt_sr, net_g, vc, if_f0, model_index)))
        categories.append([category_title, category_folder, description, models])
    with gr.Blocks() as app:
        gr.Markdown(
            "# <center> Multi Model RVC Inference\n"
            "### <center> Support v2 Model\n"
            "#### From [Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)\n"
            "[![Original Repo](https://badgen.net/badge/icon/github?icon=github&label=Original%20Repo)](https://github.com/ArkanDash/Multi-Model-RVC-Inference)"
        )
        for (folder_title, folder, description, models) in categories:
            with gr.TabItem(folder_title):
                if description:
                    gr.Markdown(f"### <center> {description}")
                with gr.Tabs():
                    if not models:
                        gr.Markdown("# <center> No Model Loaded.")
                        gr.Markdown("## <center> Please add model or fix your model path.")
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
                                with gr.Column():
                                    vc_audio_mode = gr.Dropdown(label="Input voice", choices=["Input path", "Upload audio", "Youtube", "TTS Audio"], allow_custom_value=False, value="Upload audio")
                                    # Input and Upload
                                    vc_input = gr.Textbox(label="Input audio path", visible=False)
                                    vc_upload = gr.Audio(label="Upload audio file", visible=True, interactive=True)
                                    # Youtube
                                    vc_download_audio = gr.Dropdown(label="Provider", choices=["Youtube"], allow_custom_value=False, visible=False, value="Youtube", info="Select provider (Default: Youtube)")
                                    vc_link = gr.Textbox(label="Youtube URL", visible=False, info="Example: https://www.youtube.com/watch?v=Nc0sB1Bmf-A", placeholder="https://www.youtube.com/watch?v=...")
                                    vc_split_model = gr.Dropdown(label="Splitter Model", choices=["htdemucs", "mdx_extra_q"], allow_custom_value=False, visible=False, value="htdemucs", info="Select the splitter model (Default: htdemucs)")
                                    vc_split = gr.Button("Split Audio", variant="primary", visible=False)
                                    vc_vocal_preview = gr.Audio(label="Vocal Preview", visible=False)
                                    vc_inst_preview = gr.Audio(label="Instrumental Preview", visible=False)
                                    vc_audio_preview = gr.Audio(label="Audio Preview", visible=False)
                                    # TTS
                                    tts_text = gr.Textbox(visible=False, label="TTS text", info="Text to speech input")
                                    tts_voice = gr.Dropdown(label="Edge-tts speaker", choices=voices, visible=False, allow_custom_value=False, value="en-US-AnaNeural-Female")
                                with gr.Column():
                                    spk_item = gr.Slider(
                                        minimum=0,
                                        maximum=2333,
                                        step=1,
                                        label="Speaker ID",
                                        info="(Default: 0)",
                                        value=0,
                                        interactive=True,
                                    )
                                    vc_transform0 = gr.Number(label="Transpose", value=0, info='Type "12" to change from male to female voice. Type "-12" to change female to male voice')
                                    f0method0 = gr.Radio(
                                        label="Pitch extraction algorithm",
                                        info="PM is fast, Harvest is good but extremely slow, and Crepe effect is good but requires GPU (Default: PM)",
                                        choices=["pm", "harvest", "crepe"],
                                        value="pm",
                                        interactive=True,
                                    )
                                    index_rate1 = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label="Retrieval feature ratio",
                                        info="(Default: 0.6)",
                                        value=0.6,
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
                                        value=0.35,
                                        step=0.01,
                                        interactive=True,
                                    )
                                with gr.Column():
                                    vc_log = gr.Textbox(label="Output Information", interactive=False)
                                    vc_output = gr.Audio(label="Output Audio", interactive=False)
                                    vc_convert = gr.Button("Convert", variant="primary")
                                    vc_volume = gr.Slider(
                                        minimum=0,
                                        maximum=10,
                                        label="Vocal volume",
                                        value=4,
                                        interactive=True,
                                        step=1,
                                        info="Adjust vocal volume (Default: 4}",
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
                                spk_item,
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
                        vc_split.click(
                            fn=cut_vocal_and_inst, 
                            inputs=[vc_link, vc_download_audio, vc_split_model], 
                            outputs=[vc_vocal_preview, vc_inst_preview, vc_audio_preview]
                        )
                        vc_combine.click(
                            fn=combine_vocal_and_inst,
                            inputs=[vc_output, vc_volume, vc_split_model],
                            outputs=[vc_combined_output]
                        )
                        vc_audio_mode.change(
                            fn=change_audio_mode,
                            inputs=[vc_audio_mode],
                            outputs=[
                                vc_input, 
                                vc_upload,
                                vc_download_audio,
                                vc_link,
                                vc_split_model,
                                vc_split,
                                vc_vocal_preview,
                                vc_inst_preview,
                                vc_audio_preview,
                                vc_volume,
                                vc_combined_output,
                                vc_combine,
                                tts_text,
                                tts_voice
                            ]
                        )
        app.queue(concurrency_count=1, max_size=20, api_open=config.api).launch(share=config.colab)