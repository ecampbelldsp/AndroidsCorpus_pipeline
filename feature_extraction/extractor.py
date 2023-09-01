#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  21/3/23 23:48

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

from torchvggish import vggish, vggish_input
import torch
torch.random.manual_seed(0)
import torchaudio

import opensmile
import librosa
from feature_extraction.rasta import  rastaplp
import numpy as np
import gc
from sklearn.preprocessing import MinMaxScaler
from vad.vad_sidekit import  vad_energy
import soundfile as snd
import random

class DataAugmentation():
    def __init__(self):
        pass
    def time_stretch(self,audio):
        rate = random.choice([x*0.01 for x in range(65,151)])
        stretched_audio = librosa.effects.time_stretch(audio, rate = rate)
        return stretched_audio
    def pitch_shift(self,audio, fs):
        n_steps = random.randint(-4,6)
        return librosa.effects.pitch_shift(audio, sr = fs, n_steps = n_steps)

    def add_noise(self, audio):
        noise = np.random.randn(len(audio))*0.01
        noise_factor = random.choice([x * 0.01 for x in range(50,100)])
        noisy_audio = audio + noise_factor * noise
        return noisy_audio

    def time_mask(self, audio):
        num_samples = len(audio)
        mask_factor = random.choice([x * 0.01 for x in range(20,50)])
        mask_length = int(mask_factor * num_samples)
        start = np.random.randint(0, num_samples - mask_length)
        audio[start:start + mask_length] = 0
        return audio

    def freq_mask(self, audio):
        fft = np.fft.fft(audio)
        num_bins = len(fft)
        mask_factor = random.choice([x * 0.01 for x in range(20,50)])
        mask_width = int(mask_factor * num_bins)
        start = np.random.randint(0, num_bins - mask_width)
        fft[start:start + mask_width] = 0
        audio = np.real(np.fft.ifft(fft))
        return audio

class Extractor():

    def __init__(self, feature_type = "melSpectrum", device = None, vad = False, pre = 0.97):
        self.feature_dimension = None
        self.vad = vad
        self.feature_type = feature_type
        self.device = device
        self.bundle = None
        self.scalar = MinMaxScaler()
        self.pre = pre
        self.window_size = 0.025
        self.hop = 0.01

        if self.feature_type == "melSpectrum":
            self.feature_dimension = 40
        elif self.feature_type == "egemap_func":
            self.feature_dimension = 88
        elif self.feature_type == "egemap_lld":
            self.feature_dimension = 25
        elif self.feature_type == "compare_lld":
            self.feature_dimension = 55
        elif self.feature_type == "compare_func":
            self.feature_dimension = 6373
        elif self.feature_type == "vggish":
            self.feature_dimension = 128
        elif self.feature_type == "rasta":
            self.feature_dimension = 9

        elif self.feature_type == "wav2vec2_xlsr":
            self.bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
            self.feature_dimension = 1024
            self.model = self.bundle.get_model().to(device)
        elif self.feature_type == "hubert_large":
            self.bundle = torchaudio.pipelines.HUBERT_LARGE
            self.feature_dimension = 1024
            self.model = self.bundle.get_model().to(device)
        elif self.feature_type == "wav2vec2_large":
            self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
            self.feature_dimension = 1024
            self.model = self.bundle.get_model().to(device)
        elif self.feature_type == "hubert_base":
            self.bundle = torchaudio.pipelines.HUBERT_BASE
            self.feature_dimension = 768
            self.model = self.bundle.get_model().to(device)
        elif self.feature_type == "wav2vec2_base":
            self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.feature_dimension = 768
            self.model = self.bundle.get_model().to(device)

        elif self.feature_type == "trill":
            import tensorflow as tf
            import tensorflow_hub as hub

            tf.compat.v1.enable_eager_execution()
            if str(device) == "cuda":
                gpus = tf.config.experimental.list_physical_devices('GPU')
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                except:
                    print("There is not a available GPU")

            self.feature_dimension = 512
            self.module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/2')
        else:
            raise ValueError("Invalid feature extractor ID")

    def compare_lld(self,samples, fs):

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

        feature = smile.process_signal(samples, fs).to_numpy()

        return feature

    def compare_func(self, samples, fs):

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals)

        feature = smile.process_signal(samples, fs).to_numpy()

        return feature

    def trill(self, samples, fs):
        emb_dict = self.module(samples=samples, sample_rate=fs)
        return emb_dict['embedding'].numpy()

    def egemap_func(self,samples, fs):

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals)
        feature = smile.process_signal(samples, fs).to_numpy()
        return feature

    def egemap_lld(self, samples, fs):

        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
        feature = smile.process_signal(samples, fs).to_numpy()

        return feature

    def VGGish_pipeline(self, sample, fs):
        embedding_model = vggish()
        embedding_model.eval()
        example = vggish_input.waveform_to_examples(sample, fs, return_tensor=True)
        embeddings = embedding_model.forward(example).detach().numpy()

        return embeddings

    def log_mel_fb(self, samples, fs, delta = False):
        window_size = int(self.window_size * fs)
        step = int(self.hop * fs)

        spectrogram = librosa.feature.melspectrogram(y=samples, sr=fs, fmax=fs / 2, n_fft=2048, hop_length=step,
                                                    win_length=window_size, n_mels=self.feature_dimension)

        spectrogram_log = np.log(spectrogram ** 2).transpose()

        if delta:
            delta1 = librosa.feature.delta(spectrogram_log, order=1, axis=- 1)
            delta2 = librosa.feature.delta(spectrogram_log, order=2, axis=- 1)
            spectrogram_log = np.concatenate((spectrogram_log, delta1, delta2), axis=1)

        return spectrogram_log
    def normalization_mean_variance(self, feature):
        mean = np.mean(feature, axis = 0)
        std = np.std(feature, axis = 0)

        return (feature - mean) / std

    def pre_emphasis(self,input_sig, pre):
        """Pre-emphasis of an audio signal.
        :param input_sig: the input vector of signal to pre emphasize
        :param pre: value that defines the pre-emphasis filter.
        """
        if input_sig.ndim == 1:
            return (input_sig - np.c_[input_sig[np.newaxis, :][..., :1],
            input_sig[np.newaxis, :][..., :-1]].squeeze() * pre)
        else:
            return input_sig - np.c_[input_sig[..., :1], input_sig[..., :-1]] * pre
    def vad_energy_sidekit(self, samples, fs):

        window_size = int(self.window_size * fs)
        step = int(self.hop * fs)

        frames = librosa.util.frame(samples, frame_length=window_size, hop_length=step).transpose()
        log_energy = np.sqrt(np.sum(frames ** 2, axis=1))
        label, _ = vad_energy(log_energy,
                   distrib_nb=3,
                   nb_train_it=8,
                   flooring=0.0001, ceiling=1.0,
                   alpha=1.5)

        indx = np.where(label == True)[0]
        samples_voice = []
        for i in indx:
            samples_voice.append(samples[i * window_size : i * window_size + window_size])

        # snd.write('/tmp/voice.wav', np.concatenate(samples_voice), fs)
        # snd.write('/tmp/original.wav', samples, fs)
        # try:
        #     samples_voice = np.concatenate(samples_voice)
        # except ValueError:
        #     samples_voice = np.zeros((10,))
        # if samples_voice.shape[0]/fs <= 1:
        #     print("There is not voice in the audio")
        #     return samples
        return np.concatenate(samples_voice)
    def compute(self, samples, fs, vad = True, pre_emphasis = True):
        np.random.seed(0)
        samples = samples + 0.000001 * np.random.randn(samples.shape[0])
        if pre_emphasis:
            samples = self.pre_emphasis(samples, self.pre)
        if vad:
            samples = self.vad_energy_sidekit(samples, fs)
        if self.feature_type == "melSpectrum":
            feature = self.log_mel_fb(samples, fs, delta=False)
        elif self.feature_type == "egemap_func":
            feature = self.egemap_func(samples, fs)
        elif self.feature_type == "egemap_lld":
            feature = self.egemap_lld(samples, fs)
        elif self.feature_type == "compare_func":
            feature = self.compare_func(samples, fs)
        elif self.feature_type == "compare_lld":
            feature = self.compare_lld(samples, fs)
        elif self.feature_type == "vggish":
            feature = self.VGGish_pipeline(samples, fs)
        elif self.feature_type == "rasta":
            feature = rastaplp(samples, fs=fs).transpose()
        elif self.feature_type == "trill":
            feature = self.trill(samples, fs=fs)

        elif self.feature_type == "wav2vec2_xlsr" or self.feature_type == "hubert_large" or self.feature_type == "wav2vec2_large" or self.feature_type == "hubert_base" or self.feature_type == "wav2vec2_base":
            # waveform, sample_rate = torchaudio.load(audio_path)
            waveform = torch.from_numpy(samples).unsqueeze(0).type(torch.FloatTensor)
            waveform = waveform.to(self.device)

            if fs != self.bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform, fs, self.bundle.sample_rate)

            with torch.inference_mode():
                tmp, _ = self.model.extract_features(waveform)
                feature = tmp[-1].cpu().numpy()[0, :, :]
                del tmp
                gc.collect()
                torch.cuda.empty_cache()

        else:
            raise ValueError("Invalid feature extractor ID")


        if feature.shape[0] != 1: feature = self.scalar.fit_transform(feature)

        return feature
