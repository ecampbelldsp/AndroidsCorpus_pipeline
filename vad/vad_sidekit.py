#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  18/6/23 12:50

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import numpy
from scipy import ndimage
import copy
from vad.mixture import Mixture

def vad_energy(log_energy,
               distrib_nb=3,
               nb_train_it=8,   #nb_train_it=8,
               flooring=0.0001, ceiling=1.0,  #flooring=0.0001, ceiling=1.0,
               alpha=2):  #alpha=2
    """

    :param log_energy:
    :param distrib_nb:
    :param nb_train_it:
    :param flooring:
    :param ceiling:
    :param alpha:
    :return:
    """
    # center and normalize the energy
    log_energy = (log_energy - numpy.mean(log_energy)) / numpy.std(log_energy)

    # Initialize a Mixture with 2 or 3 distributions
    world = Mixture()
    # set the covariance of each component to 1.0 and the mean to mu + meanIncrement
    world.cst = numpy.ones(distrib_nb) / (numpy.pi / 2.0)
    world.det = numpy.ones(distrib_nb)
    world.mu = -2 + 4.0 * numpy.arange(distrib_nb) / (distrib_nb - 1)
    world.mu = world.mu[:, numpy.newaxis]
    world.invcov = numpy.ones((distrib_nb, 1))
    # set equal weights for each component
    world.w = numpy.ones(distrib_nb) / distrib_nb
    world.cov_var_ctl = copy.deepcopy(world.invcov)

    # Initialize the accumulator
    accum = copy.deepcopy(world)

    # Perform nbTrainIt iterations of EM
    for it in range(nb_train_it):
        accum._reset()
        # E-step
        world._expectation(accum, log_energy)
        # M-step
        world._maximization(accum, ceiling, flooring)

    # Compute threshold
    threshold = world.mu.max() - alpha * numpy.sqrt(1.0 / world.invcov[world.mu.argmax(), 0])

    # Apply frame selection with the current threshold
    label = log_energy > threshold
    return label, threshold


def label_fusion(label, win=3):
    """Apply a morphological filtering on the label to remove isolated labels.
    In case the input is a two channel label (2D ndarray of boolean of same
    length) the labels of two channels are fused to remove
    overlaping segments of speech.

    :param label: input labels given in a 1D or 2D ndarray
    :param win: parameter or the morphological filters
    """
    channel_nb = len(label)
    if channel_nb == 2:
        overlap_label = numpy.logical_and(label[0], label[1])
        label[0] = numpy.logical_and(label[0], ~overlap_label)
        label[1] = numpy.logical_and(label[1], ~overlap_label)

    for idx, lbl in enumerate(label):
        cl = ndimage.grey_closing(lbl, size=win)
        label[idx] = ndimage.grey_opening(cl, size=win)

    return label