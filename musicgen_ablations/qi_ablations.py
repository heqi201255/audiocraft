import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from audiocraft.models import MusicGen
import torchaudio
from audiocraft.models.section_patterns import *

model = MusicGen.get_pretrained('facebook/musicgen-large')

def generate(sec_pattern: str, key: str, demo_id: int):
    prompt = f"Generate an electronic pop track at 120BPM in {key} major. Follow this structure: {form_prompt_structure(sec_pattern)}."
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=total_bars(sec_pattern),
        extend_stride=10
    )
    output = model.generate(descriptions=[prompt], progress=True, return_tokens=False, sec_pattern=sec_pattern)
    save_path = f"/root/audiocraft/musicgen_ablations/outputs/{sec_pattern}"
    os.makedirs(save_path, exist_ok=True)
    torchaudio.save(os.path.join(save_path, f"musicgen_{sec_pattern}_{key}_{demo_id}.wav"), output[0], 32000)

# def generate_continuation(text_prompt: str, audio_file: str):
#     prompt_waveform, prompt_sr = torchaudio.load(audio_file)
#     prompt_duration = 20
#     prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
#     output = model.generate_continuation(prompt_waveform, descriptions=[text_prompt], prompt_sample_rate=prompt_sr, progress=True, return_tokens=False)
#     torchaudio.save("/root/audiocraft/musicgen_ablations/outputs/continuation_demo.mp3", output[0].detach().cpu(), 32000)
#
# prompt = f"Generate an electronic pop track at 120BPM in C major. Follow this structure: {}"
#
# generate_continuation(prompt, "/root/audiocraft/musicgen_ablations/song_for_continuation/demo2.mp3")
for sp in ('pattern1', 'pattern2', 'pattern3', 'pattern4'):
    for key in ('C', 'F', 'G', 'A#'):
        for i in range(2):
            generate(sp, key, i+1)