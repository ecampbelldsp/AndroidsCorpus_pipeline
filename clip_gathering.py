#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  31/8/23 23:35

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

import numpy as np
import os
import soundfile as snd

CORPUS = "Androids-Corpus"
TASK = "Interview-Task"
SAVE = f"/media/ecampbell/D/Data-io/{CORPUS}/{TASK}/audio_clip_gathering/"


label = np.genfromtxt(f"label/label_{CORPUS}_{TASK}.txt", dtype=str)
        # for task in ["Interview-Task", "Reading-Task"]:
if TASK == "Interview-Task":
    audio_path = f"/media/ecampbell/D/Data-io/Androids-Corpus/{TASK}/audio_clip/"
elif TASK == "Reading-Task":
    audio_path = f"/media/ecampbell/D/Data-io/Androids-Corpus/{TASK}/audio/"
duration_pos, duration_neg = [], []
for i, name in enumerate(label[1:,0]):
    print(f"Recording {i+1} - {len(label[1:,0])}  {TASK}")

    name = name[:-4]
    audio_clips = os.listdir(f"{audio_path}{name}")
    samples = []
    for name_clips in audio_clips:
        samples_clip,fs = snd.read(f"{audio_path}{name}/{name_clips}")
        samples.append(samples_clip)
    samples = np.concatenate(samples)
    snd.write(f"{SAVE}{name}.wav",samples,fs)
    if np.any(np.isnan(samples)):
        raise ValueError(f"Signal {name} has NAN values")



