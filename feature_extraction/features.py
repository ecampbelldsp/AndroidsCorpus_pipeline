#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  2/3/23 11:45

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import opensmile
import librosa
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torchvggish


def egemap(samples, fs):
    window_size = int(0.025 * fs)
    step = int(0.01 * fs)

    np.random.seed(0)
    samples = samples + 0.000001 * np.random.randn(samples.shape[0])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feature = smile.process_signal(samples, fs).to_numpy()

    # scalar = MinMaxScaler()
    # feature = scalar.fit_transform(feature)

    return feature


def egemap_lld(samples, fs):
    window_size = int(0.025 * fs)
    step = int(0.01 * fs)

    np.random.seed(0)
    samples = samples + 0.000001 * np.random.randn(samples.shape[0])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    feature = smile.process_signal(samples, fs).to_numpy()

    # scalar = MinMaxScaler()
    # feature = scalar.fit_transform(feature)

    return feature


def ComParE_lld(samples, fs):
    window_size = int(0.025 * fs)
    step = int(0.01 * fs)

    np.random.seed(0)
    samples = samples + 0.000001 * np.random.randn(samples.shape[0])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    feature = smile.process_signal(samples, fs).to_numpy()

    # scalar = MinMaxScaler()
    # feature = scalar.fit_transform(feature)

    return feature

def ComParE_func(samples, fs):
    window_size = int(0.025 * fs)
    step = int(0.01 * fs)

    np.random.seed(0)
    samples = samples + 0.000001 * np.random.randn(samples.shape[0])

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feature = smile.process_signal(samples, fs).to_numpy()

    # scalar = MinMaxScaler()
    # feature = scalar.fit_transform(feature)

    return feature

def VGGish_pipeline(audio_path, embedding_model, feat):
    example = torchvggish.vggish_input.wavfile_to_examples(audio_path)
    embeddings = embedding_model.forward(example).detach().numpy()

    if feat == "vggish":
        embeddings = np.mean(embeddings, axis=0)[np.newaxis, :]

    return embeddings


def log_mel_fb(samples, fs,feature_dimension):
    window_size = int(0.025 * fs)
    step = int(0.01 * fs)

    np.random.seed(0)
    samples = samples + 0.000001 * np.random.randn(samples.shape[0])
    spectogram = librosa.feature.melspectrogram(y=samples, sr=fs, fmax=fs / 2, n_fft=2048, hop_length=step,
                                                win_length=window_size, n_mels=feature_dimension)
    spectogram_log = np.log(spectogram ** 2).transpose()
    # delta1 = librosa.feature.delta(spectogram_log, width=9, order=1, axis=- 1)
    # delta2 = librosa.feature.delta(spectogram_log, width=9, order=2, axis=- 1)

    # feature = np.concatenate((spectogram_log,delta1,delta2),axis=0).transpose()

    scalar = MinMaxScaler()
    feature = scalar.fit_transform(spectogram_log)

    return feature