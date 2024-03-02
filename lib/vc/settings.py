import gradio as gr

def change_audio_mode(vc_audio_mode):
    if vc_audio_mode == "Input path":
        return (
            # Input & Upload
            gr.Textbox(visible=True),
            gr.Audio(visible=False),
            # Youtube
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            # Splitter
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Audio(visible=False),
            gr.Button(visible=False),
            # TTS
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False)
        )
    elif vc_audio_mode == "Upload audio":
        return (
            # Input & Upload
            gr.Textbox(visible=False),
            gr.Audio(visible=True),
            # Youtube
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            # Splitter
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Audio(visible=False),
            gr.Button(visible=False),
            # TTS
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False)
        )
    elif vc_audio_mode == "Youtube":
        return (
            # Input & Upload
            gr.Textbox(visible=False),
            gr.Audio(visible=False),
            # Youtube
            gr.Dropdown(visible=True),
            gr.Textbox(visible=True),
            gr.Textbox(visible=True),
            gr.Button(visible=True),
            # Splitter
            gr.Dropdown(visible=True),
            gr.Textbox(visible=True),
            gr.Button(visible=True),
            gr.Audio(visible=True),
            gr.Audio(visible=True),
            gr.Audio(visible=True),
            gr.Slider(visible=True),
            gr.Slider(visible=True),
            gr.Audio(visible=True),
            gr.Button(visible=True),
            # TTS
            gr.Textbox(visible=False),
            gr.Dropdown(visible=False)
        )
    elif vc_audio_mode == "TTS Audio":
        return (
            # Input & Upload
            gr.Textbox(visible=False),
            gr.Audio(visible=False),
            # Youtube
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            # Splitter
            gr.Dropdown(visible=False),
            gr.Textbox(visible=False),
            gr.Button(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Audio(visible=False),
            gr.Slider(visible=False),
            gr.Slider(visible=False),
            gr.Audio(visible=False),
            gr.Button(visible=False),
            # TTS
            gr.Textbox(visible=True),
            gr.Dropdown(visible=True)
        )