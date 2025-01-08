import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from audiocraft.models import MusicGen
import torchaudio
# from audiocraft.utils.notebook import display_audio

model = MusicGen.get_pretrained('facebook/musicgen-large')
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=120,
    extend_stride=10
)

def generate(text_prompt: str):
    output = model.generate(descriptions=[text_prompt], progress=True, return_tokens=False)
    torchaudio.save("/root/audiocraft/musicgen_ablations/outputs/generated_demo.mp3", output[0], 32000)
    # display_audio(output, sample_rate=32000)

def generate_continuation(text_prompt: str, audio_file: str):
    prompt_waveform, prompt_sr = torchaudio.load(audio_file)
    prompt_duration = 20
    prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
    output = model.generate_continuation(prompt_waveform, descriptions=[text_prompt], prompt_sample_rate=prompt_sr, progress=True, return_tokens=False)
    torchaudio.save("/root/audiocraft/musicgen_ablations/outputs/continuation_demo.mp3", output[0].detach().cpu(), 32000)
    # display_audio(output, sample_rate=32000)

prompt = "Generate an electronic pop track at 120BPM in C major, happy mood. Follow this structure: 4-bar intro, 8-bar verse, 4-bar pre-chorus, 8-bar chorus, 8-bar verse, 4-bar pre-chorus, 8-bar chorus, 4-bar bridge, 8-bar chorus, 4-bar outro."

generate_continuation(prompt, "/root/audiocraft/musicgen_ablations/song_for_continuation/demo2.mp3")