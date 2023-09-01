#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  1/9/23 18:59

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""

FEATURE_PARMS = {
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