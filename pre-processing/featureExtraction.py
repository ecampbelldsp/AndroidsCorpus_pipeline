#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  29/8/23 9:08

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""


import os
import sys
import numpy as np
import h5py
import soundfile as snd
import librosa
import torch

from feature_extraction.extractor import Extractor,DataAugmentation

import multiprocessing


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
def extract_features(label, path, hf, feature_extractor, data_augmentation = False,cut_start_and_end = False, set = None, task = None):
        for i, name in enumerate(label[:, 0]):
            print(f"{task.upper()} {set.upper()} {i + 1} - {label.shape[0]} {feature_extractor.feature_type}")
            try:

                samples, fs = snd.read(f"{path}{name}")
                duration = len(samples) / fs

                if cut_start_and_end: samples = samples[int(fs*5):int(-fs*5)]
                if fs != 16000:
                    samples = librosa.resample(samples, orig_sr=fs, target_sr=16000)

                    fs = 16000
                samples = feature_extractor.vad_energy_sidekit(samples, fs)
                feature = feature_extractor.compute(samples, fs,vad = False, pre_emphasis = True)
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
                        feature_augmentation[technique] = feature_extractor.compute(samples_augmentation[technique], fs,vad = False, pre_emphasis = True)

            except RuntimeError:
                raise ValueError("Found corrupted audio")
            try:
                hf.create_dataset(name, data=feature)
                if data_augmentation:
                    for technique in feature_augmentation:
                        hf.create_dataset(name + "_" + technique, data=feature_augmentation[technique])
            except RuntimeError:
                raise ValueError("name already exists")
        hf.close()

def main(feature_type, device, set ="KCL"):
    if device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        "compare_lld": (65,500)
    }

    try:
        os.mkdir("features")
    except:
        pass
    for task in ["Interview-Task"]:
        feature_extractor = Extractor(feature_type=feature_type, device=device, vad =True)
        path = {"server":f"/media/ecampbell/D/Data-io/Androids-Corpus/{set}/{task}/audio/",
                "local": f"/media/ecampbell/D/Data-io/Androids-Corpus/{set}/{task}/audio/"}#"/home/ecampbell/Storage/coding/Data-io/depression_2021-AUDIOS/"

        label = np.genfromtxt(f"label/label_AndroidsCorpus.txt", dtype=str, delimiter=" ")

        hf = h5py.File(f"features/{feature_type}_{set}_{task}.h5", 'w')
        extract_features(label, path["local"], hf, feature_extractor, data_augmentation=False,cut_start_and_end = False, set = set, task = task)


if __name__ == '__main__':

        device = "gpu"#"cpu"
        paralel = False

        feature_list = ["melSpectrum"]


        if paralel:
            pool = multiprocessing.Pool(len(feature_list))
            processes = [pool.apply_async(main, args=(feature_type,device, "Androids-Corpus")) for feature_type in feature_list]
            result = [p.get() for p in processes]
        else:
            for feature_type in feature_list:
                main(feature_type, device, "Androids-Corpus")
