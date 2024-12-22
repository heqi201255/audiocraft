# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from .encodec import CompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition, StyleConditioner


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
    "style": "facebook/musicgen-style",
}


import math
from typing import Union


class BarStepTick:
    """
    This clss is used as the DAW time unit in many other modules. From its name you can tell it means the bar, step, and
    tick typically used in modern DAWs. In MIDI standards, it only uses Ticks for timing, you'll need to convert it
    to other timing units with stuff like PPQ. Here we use BarStepTick as the only time unit across Sprin to simplify
    the understanding of music timing.
    """
    def __init__(self, bar: Union[int, list] = 0, step: int = 0, tick: int = 0):
        """
        The creation of a BarStepTick instance, it will check all the params whether they are in valid ranges.
        :param bar: should be a non-negative integer. Each bar has 16 steps.
        :param step: should be between 0 and 15 inclusively. Each step has 24 ticks.
        :param tick: should be between 0 and 23 inclusively.
        """
        if type(bar) is list:
            if len(bar) == 1:
                bbar = int(bar[0])
                bstep = 0
                btick = 0
            elif len(bar) == 2:
                bbar = int(bar[0])
                bstep = int(bar[1])
                btick = 0
            elif len(bar) == 3:
                bbar = int(bar[0])
                bstep = int(bar[1])
                btick = int(bar[2])
            else:
                raise ValueError("Wrong format")
        else:
            bbar = int(bar)
            bstep = int(step)
            btick = int(tick)
        if bbar < 0:
            raise ValueError("'Bar' must be a non-negative integer")
        elif bstep < 0 or bstep > 15:
            raise ValueError("'Step' must be between 0 and 15 inclusively")
        elif btick < 0 or btick > 23:
            raise ValueError("'Tick' must be between 0 and 23 inclusively")
        self.bar = bbar
        self.step = bstep
        self.tick = btick

    def __hash__(self):
        return hash(self.to_steps())

    def get_bar_step_tick(self):
        return self.bar, self.step, self.tick

    def append(self, bst: Union['BarStepTick', list[int, int], list[int, int, int]], inplace=False):
        if type(bst) is BarStepTick:
            bar = self.bar + bst.bar
            step = self.step + bst.step
            tick = self.tick + bst.tick
            bar += int((step + int(tick / 24)) / 16)
            step = (step + int(tick / 24)) % 16
            tick = tick % 24
        else:
            if len(bst) == 2:
                bar = self.bar + bst[0]
                step = self.step + bst[1]
                bar += int(step / 16)
                step = step % 16
                tick = self.tick
            elif len(bst) == 3:
                bar = self.bar + bst[0]
                step = self.step + bst[1]
                tick = self.tick + bst[2]
                bar += int((step + int(tick / 24)) / 16)
                step = (step + int(tick / 24)) % 16
                tick = tick % 24
            else:
                raise ValueError("'bst' should be a 'BarStepTick' instance or a list of two integers or three integers.")
        if inplace:
            self.bar = bar
            self.step = step
            self.tick = tick
        return bar, step, tick

    def __lt__(self, other):
        if self.to_ticks() < other.to_ticks():
            return True
        return False

    def __eq__(self, other):
        if self.to_ticks() == other.to_ticks():
            return True
        return False

    def __le__(self, other):
        if self.to_ticks() <= other.to_ticks():
            return True
        return False

    def __add__(self, other):
        return BarStepTick(*self.append(other))

    def __sub__(self, other):
        if other.bar < self.bar:
            if other.step <= self.step:
                if other.tick <= self.tick:
                    bar = self.bar - other.bar
                    step = self.step - other.step
                    tick = self.tick - other.tick
                else:  # [1,1,15] [0, 1, 20]
                    tick = 24 - (other.tick - self.tick)
                    step = self.step - other.step - 1
                    if step < 0:
                        step += 16
                        bar = self.bar - other.bar - 1
                    else:
                        bar = self.bar - other.bar
            else:
                if other.tick <= self.tick:
                    bar = self.bar - other.bar - 1
                    step = 16 - (other.step - self.step)
                    tick = self.tick - other.tick
                else:  # [2,1,15] [0,15,20]
                    tick = 24 - (other.tick - self.tick)
                    step = 16 - (other.step - self.step) - 1
                    bar = self.bar - other.bar - 1
        elif other.bar == self.bar:
            bar = 0
            if other.step <= self.step:
                if other.tick <= self.tick:
                    step = self.step - other.step
                    tick = self.tick - other.tick
                else:
                    tick = 24 - (other.tick - self.tick)
                    step = self.step - other.step - 1
                    if step < 0:
                        raise ValueError("BarStepTick cannot subtract another BarStepTick instance that is longer.")
            else:
                raise ValueError("BarStepTick cannot subtract another BarStepTick instance that is longer.")
        else:
            raise ValueError("BarStepTick cannot subtract another BarStepTick instance that has more length.")
        return BarStepTick(bar, step, tick)

    def is_empty(self):
        return self.bar == 0 and self.step == 0 and self.tick == 0

    def to_beats(self):
        beat = self.bar * 4
        beat += self.step / 4
        beat += self.tick / 96
        return round(beat, 2)

    def to_bars(self):
        return round(self.bar + (self.step / 16) + (self.tick / 384), 2)

    def to_seconds(self, bpm):
        return BarStepTick.bst2sec(self, bpm=bpm)

    def to_steps(self):
        return self.bar * 16 + self.step + round(self.tick / 24)

    def to_ticks(self):
        return self.bar * 384 + self.step * 24 + self.tick

    @staticmethod
    def str2bst(s: str) -> 'BarStepTick':
        try:
            bind = s.index("b")
        except:
            bind = 0
        try:
            sind = s.index("s")
        except:
            sind = 0
        try:
            tind = s.index("t")
        except:
            tind = 0
        bar = step = tick = 0
        if bind:
            bar = int(s[:bind])
        if sind:
            if bind:
                step = int(s[bind + 1:sind])
            else:
                step = int(s[:sind])
        if tind:
            if sind:
                tick = int(s[sind + 1:tind])
            elif bind:
                tick = int(s[bind + 1:tind])
            else:
                tick = int(s[:tind])
        return BarStepTick(bar, step, tick)

    def __str__(self):
        s = []
        if self.bar != 0:
            s.append(str(self.bar) + "b")
        if self.step != 0:
            s.append(str(self.step) + "s")
        if self.tick != 0:
            s.append(str(self.tick) + "t")
        if not s:
            s = ["0b"]
        return "".join(s)

    def __repr__(self):
        return self.__str__()

    def get_bar(self):
        return self.bar

    def get_step(self):
        return self.step

    def get_tick(self):
        return self.tick

    def set_bar(self, bar: int):
        bar = int(bar)
        if bar < 0:
            raise ValueError("'Bar' must be a non-negative integer")
        self.bar = bar

    def set_step(self, step: int):
        step = int(step)
        if step < 0 or step > 15:
            raise ValueError("'Step' must be between 0 and 15 inclusively")
        self.step = step

    def set_tick(self, tick: int):
        tick = int(tick)
        if tick < 0 or tick > 23:
            raise ValueError("'Tick' must be between 0 and 23 inclusively")
        self.tick = tick

    @staticmethod
    def sec2bst(sec: float, bpm: float = 120) -> 'BarStepTick':
        '''
        BST is the Bar-Step-Tick time signature, but in pretty_midi we need the exact start time and end time in seconds to
        draw a note, so we need this function to convert the BST to seconds given the bpm, bar, and step. Tick is not used
        here because it is too small and we really don't need it.
        '''
        total_ticks = math.ceil(sec / (60 / (bpm * 96)))
        ticks = total_ticks % 24
        steps = int(total_ticks / 24)
        bar = int(steps / 16)
        steps = steps % 16
        # bar += 1
        # steps += 1
        return BarStepTick(bar, steps, ticks)

    @staticmethod
    def bst2sec(bst: 'BarStepTick', bpm: float = 120) -> float:
        '''
        BST is the Bar-Step-Tick time signature, but in pretty_midi we need the exact start time and end time in seconds
        to draw a note, so we need this function to convert the BST to seconds given the bpm, bar, and step.
        '''
        barsec = bst.bar * (60 / (bpm / 4))
        stepsec = bst.step * (60 / (bpm / 4)) / 16
        ticksec = 1 / 192 * bst.tick
        return barsec + stepsec + ticksec

    @staticmethod
    def beat2sec(beat: float, bpm) -> float:
        # step = int(beat * 4)
        # return BarStepTick.bst2sec(BarStepTick.step2bst(step), bpm=bpm)
        return beat * (60 / (bpm / 4)) / 4

    @staticmethod
    def step2bst(step: int) -> 'BarStepTick':
        '''
        Calculate the bar and step location based on steps, this function is used when implement grooves.
        '''
        bar = math.floor(step / 16)
        step = (step - (bar * 16)) % 16
        #     print(bar,step)
        return BarStepTick(bar, step)

    @staticmethod
    def beat2steps(beat: float) -> int:
        steps = round(beat * 4)
        return steps

    @staticmethod
    def bst2beats(bst: 'BarStepTick'):
        return bst.bar * 4 + bst.step / 4 + bst.tick / 24

    @staticmethod
    def sec2beats(sec: float, bpm: int = 120) -> float:
        bst = BarStepTick.sec2bst(sec, bpm)
        return bst.bar * 4 + bst.step / 4

    @staticmethod
    def sec2bars(sec: float, bpm: float = 120) -> float:
        return math.ceil(sec / (60 / (bpm * 96))) / 384

class MusicGen(BaseGenModel):
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default duration

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """Return pretrained model, we provide four models:
        - facebook/musicgen-small (300M), text to music,
          # see: https://huggingface.co/facebook/musicgen-small
        - facebook/musicgen-medium (1.5B), text to music,
          # see: https://huggingface.co/facebook/musicgen-medium
        - facebook/musicgen-melody (1.5B) text to music and text+melody to music,
          # see: https://huggingface.co/facebook/musicgen-melody
        - facebook/musicgen-large (3.3B), text to music,
          # see: https://huggingface.co/facebook/musicgen-large
        - facebook/musicgen-style (1.5 B), text and style to music,
          # see: https://huggingface.co/facebook/musicgen-style
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm, max_duration=30)

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        lm = load_lm_model(name, device=device)
        compression_model = load_compression_model(name, device=device)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
            lm.condition_provider.conditioners['self_wav']._use_masking = False

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              cfg_coef_beta: tp.Optional[float] = None,
                              two_step_cfg: bool = False, extend_stride: float = 18,):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            cfg_coef_beta (float, optional): beta coefficient in double classifier free guidance.
                Should be only used for MusicGen melody if we want to push the text condition more than
                the audio conditioning. See paragraph 4.3 in https://arxiv.org/pdf/2407.12563 to understand
                double CFG.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
            'cfg_coef_beta': cfg_coef_beta,
        }

    def set_style_conditioner_params(self, eval_q: int = 3, excerpt_length: float = 3.0,
                                     ds_factor: tp.Optional[int] = None,
                                     encodec_n_q: tp.Optional[int] = None) -> None:
        """Set the parameters of the style conditioner
        Args:
            eval_q (int): the number of residual quantization streams used to quantize the style condition
                the smaller it is, the narrower is the information bottleneck
            excerpt_length (float): the excerpt length in seconds that is extracted from the audio
                conditioning
            ds_factor: (int): the downsampling factor used to downsample the style tokens before
                using them as a prefix
            encodec_n_q: (int, optional): if encodec is used as a feature extractor, sets the number
                of streams that is used to extract features
        """
        assert isinstance(self.lm.condition_provider.conditioners.self_wav, StyleConditioner), \
            "Only use this function if you model is MusicGen-Style"
        self.lm.condition_provider.conditioners.self_wav.set_params(eval_q=eval_q,
                                                                    excerpt_length=excerpt_length,
                                                                    ds_factor=ds_factor,
                                                                    encodec_n_q=encodec_n_q)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)
            x = 0
            sections = f"{'i' * 4}{'v' * 8}{'p' * 4}{'c' * 8}{'v' * 8}{'p' * 4}{'c' * 8}{'b' * 4}{'c' * 8}{'o' * 4}"
            sec_convert = {'i': 'Intro', 'v': 'Verse', 'p': 'Pre-Chorus', 'c': 'Chorus', 'b': 'Bridge', 'o': 'Outro'}
            while current_gen_offset + prompt_length < total_gen_len:
                print(f"Iteration {x}")
                x += 1
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                print(time_offset, chunk_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                from_sec = time_offset
                to_sec = from_sec + chunk_duration
                from_bar = int(BarStepTick.sec2bars(from_sec, 120))
                to_bar = int(BarStepTick.sec2bars(to_sec, 120))
                belong_sections = sections[from_bar: to_bar]
                sections_stat = []
                for s in belong_sections:
                    label = sec_convert[s]
                    if not sections_stat:
                        sections_stat.append([label, 1])
                        continue
                    if sections_stat[-1][0] == label:
                        sections_stat[-1][1] = sections_stat[-1][1]+1
                    else:
                        sections_stat.append([label, 1])
                sections_stat = [f"{count} bars of {label}" for label, count in sections_stat]
                if len(sections_stat) > 1:
                    sections_stat[-1] = "and " + sections_stat[-1]
                sections_stat = ", ".join(sections_stat)
                for ii, att in enumerate(attributes):
                    prompt_with_detail = (f"{att.text['description']}\nFor now, you are generating the segment "
                                               f"between {from_sec}th second and {to_sec}th second, which corresponds to "
                                               f"the {from_bar}th bar and {to_bar}th bar regarding to the whole song, "
                                               f"your generated segment includes {sections_stat} of the song structure.")
                    att.text['description'] = prompt_with_detail
                    print(f"Iteration prompt: {att.text['description']}")
                    print(f"Attribute text: {att.text}\nAttribute wav: {att.wav}\nAttribute attributes: {att.attributes}\nText attributes: {att.text_attributes}\nWav attributes: {att.wav_attributes}\nJoint embed: {att.joint_embed}\nJoint embed attributes: {att.joint_embed_attributes}")
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens
