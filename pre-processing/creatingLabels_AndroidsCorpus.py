#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  29/8/23 8:26

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import numpy as np
import os

task_list = ["Interview-Task", "Reading-Task"]

for task in task_list:

    audioPath = f"/media/ecampbell/D/Data-io/Androids-Corpus/{task}/audio/"

    with open(f"label_Androids-Corpus_{task}.txt", "w") as file:
        file.write("ID  GENDER  AGE  EDUCATIONAL-LEVEL  CONDITION\n")
        for condition in ["HC","PT"]:
            samples = os.listdir(f"{audioPath}{condition}/")
            for name in samples:
                age = name[5:7]
                gender = name[4]
                edu = name[8]
                state = 0 if condition == "HC" else 1
                file.write(f"{name}  {gender}  {age}  {edu}  {state}\n")

a= 1