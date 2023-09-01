#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 9:24

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import numpy as np
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os

os.chdir("../")
SAVE = "default-folds_Androids-Corpus"
if not os.path.exists("default-folds_Androids-Corpus"):
    os.mkdir(SAVE)
pd_folds = pd.read_csv("fold-lists_Androids-Corpus.csv", header = None)

TASK_LIST = ["Interview-Task", "Reading-Task"]  #


for task in TASK_LIST:
    folds_list = [[],[],[],[],[]]
    if task == "Interview-Task":
        pointer = 7
    elif task == "Reading-Task":
        pointer = 0
    with open(f"{SAVE}/fold_{task}.txt","w") as file:
        for i in range(5):
            fold = pd_folds[pointer+i].values
            for value in fold:
                if type(value) is str and len(value) == 11:
                    # if len(value) == 11:
                        folds_list[i].append(value)
                        file.write(value.replace("'","")+".wa ")
            file.write("\n")
a = 1