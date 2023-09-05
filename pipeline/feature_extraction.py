#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  29/8/23 9:08

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import os
os.chdir("../")

import time
import numpy as np

import h5py
import soundfile as snd
import torch
import multiprocessing
import librosa

from FeatureExtraction.extractor import Extractor,DataAugmentation
from box.utilities import select_gpu_with_most_free_memory

start = time.time()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

feature_params = {
    "hubert_base": (768, 250),
    "wav2vec2_base": (768, 250),
    "hubert_large": (1024, 250),
    "wav2vec2_large": (1024, 250),
    "wav2vec2_xlsr": (1024, 250),
    "trill": (512, 25),
    "vggish": (128, 5),
    "melSpectrum": (40, 500),
    "rasta": (9, 500),
    "egemap_lld": (25, 500),
    "egemap_func": (88, 1),
    "compare_func": (6373, 1),
    "compare_lld": (65, 500)
}
def get_features_per_chunk(feature_extractor,samples, fs, duration, audio_max_length, PREEMPHASIS, VAD):
    quotient = int(duration // audio_max_length)
    feature = []
    for j in range(quotient):
        if j != quotient - 1:
            samples_cut = samples[int(j * audio_max_length * fs):int((j + 1) * audio_max_length * fs)]
            feature.append(feature_extractor.compute(samples_cut, fs, vad=VAD, pre_emphasis=PREEMPHASIS))
        else:
            samples_cut = samples[int(j * audio_max_length * fs):]
            if len(samples_cut) / fs > 5:
                feature.append(feature_extractor.compute(samples_cut, fs, vad=VAD, pre_emphasis=PREEMPHASIS))

    return np.concatenate(feature)
def extract_features(label, path, hf, feature_extractor, data_augmentation = False, set = None, task = None):

        VAD = False
        PREEMPHASIS = True
        for i, name in enumerate(label[1:, 0]):
            print(f"{task.upper()} {set.upper()} {i + 1} - {label.shape[0]-1} {feature_extractor.feature_type}")

            samples, fs = snd.read(f"{path}{name}")
            duration = len(samples) / fs
            audio_max_length = 30

            if fs != 16000:
                samples = librosa.resample(samples, orig_sr=fs, target_sr=16000)
                fs = 16000

            if duration > audio_max_length:
                feature = get_features_per_chunk(feature_extractor,samples, fs, duration, audio_max_length, PREEMPHASIS, VAD)
            else:
                feature = feature_extractor.compute(samples, fs,vad = VAD, pre_emphasis = PREEMPHASIS)

            samples_augmentation = {"noisy": "","time_stretch": "","pitch_shift": "","time_mask": ""}
            if data_augmentation:
                augmentation = DataAugmentation()
                samples_augmentation["noisy"] = augmentation.add_noise(samples)
                samples_augmentation["time_stretch"] = augmentation.time_stretch(samples)
                samples_augmentation["pitch_shift"] = augmentation.pitch_shift(samples, fs)
                samples_augmentation["time_mask"] = augmentation.time_mask(samples)
                # samples_augmentation["freq_mask"] = augmentation.freq_mask(samples)
            # for technique in samples_augmentation:
            #     name = name.replace("/", "_")
            #     snd.write(f"/tmp/{name[:-4]}_{technique}.wav", samples_augmentation[technique], fs)
                feature_augmentation = {"noisy": "","time_stretch": "","pitch_shift": "","time_mask": ""}
                for technique in samples_augmentation:
                    if duration > audio_max_length:
                        feature_augmentation[technique] = get_features_per_chunk(feature_extractor, samples_augmentation[technique], fs, duration, audio_max_length,
                                                         PREEMPHASIS, VAD)
                    else:
                        feature_augmentation[technique] = feature_extractor.compute(samples_augmentation[technique], fs, vad=VAD, pre_emphasis=PREEMPHASIS)
                    # feature_augmentation[technique] = feature_extractor.compute(samples_augmentation[technique], fs,vad = VAD, pre_emphasis = PREEMPHASIS)

            if np.any(np.isnan(feature)):
                raise ValueError(f"Signal {name} has NAN values")
            hf.create_dataset(name, data=feature)

            if data_augmentation:
                for technique in feature_augmentation:
                    if np.any(np.isnan(feature_augmentation[technique])):
                        raise ValueError(f"Data augmentation application ( technique {technique}) of signal {name} has NAN values")
                    hf.create_dataset(name + "_" + technique, data=feature_augmentation[technique])
        hf.close()

def main(feature_type, set ="", TASK_LIST = "", root = "", with_interviewer = False):
    selected_device, device =select_gpu_with_most_free_memory()
    # if device == "gpu":
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("../features"):
        os.mkdir("../features")

    for task in TASK_LIST:
        feature_extractor = Extractor(feature_type=feature_type, device=device, vad =True)
        label = np.genfromtxt(f"label/label_{set}_{task}.txt", dtype=str, delimiter=" ")

        if with_interviewer:
            audio_path = f"{root}{set}/{task}/audio/"
        else:
            audio_path = f"{root}{set}/{task}/audio_clip_gathering/"  #audio
        if not os.path.exists("features"):
            os.mkdir("features")
        hf = h5py.File(f"features/{feature_type}_{set}_{task}.h5", 'w')
        extract_features(label, audio_path, hf, feature_extractor, data_augmentation=False, set = set, task = task)


if __name__ == '__main__':

        paralel = False

        CORPUS = "Androids-Corpus"
        TASK_LIST = ["Interview-Task"]
        root = "/media/ecampbell/D/Data-io/"# "Reading-Task"
        feature_list = ["wav2vec2_base", "rasta", "melSpectrum","egemap_lld", "compare_lld", "hubert_base"]
        with_interviewer = True


        if paralel:
            pool = multiprocessing.Pool(len(feature_list))
            processes = [pool.apply_async(main, args=(feature_type, CORPUS, TASK_LIST,root,with_interviewer)) for feature_type in feature_list]
            result = [p.get() for p in processes]
        else:
            for feature_type in feature_list:
                main(feature_type, CORPUS, TASK_LIST,root,with_interviewer)

        print(f"Execution time: {(time.time() - start) / 3600} hours")