"""
    This software is Copyright by the Board of Trustees of Michigan
    State University (c) Copyright 2017.

    You may use this software under the terms of the GNU public license
    (GPL).  The terms of this license are described at:

     http://www.gnu.org/licenses/gpl.txt

     Authors:
             Ron Fox
             Jeromy Tompkins
             NSCL
             Michigan State University
             East Lansing, MI 48824-1321
"""


import math


def logistic(A, k, x1, x):
    """
    Evaluate a logistic function for the specified parameters and point. A logistic function
    is a function with a sigmoidal shape. We use it to fit the rising edge of signals DDAS digitizes
    from the detectors.
    :param A: Amplitude of the signal
    :param k: steepness of the signal (related to the rise time)
    :param x1: Mid point of the rise of the sigmoid
    :param x: Location at which to evaluate the function
    :return:
    """
    return A/(1+math.exp(-k*(x-x1)))


def decay(A, k, x1, x):
    """
    Signals from detectors usually have a falling shape that approximates and exponential.
    This function evaluates this decay at some point.

    :param A: amplitude of the signal
    :param k: Decay time factor of the signal
    :param x1: Position of the pulse
    :param x: where to evaluate the signal
    :return:
    """
    return A*(math.exp(-k*(x-x1)))


def singlePulse(A1, k1, k2, x1, C,  x):
    """
    Evaluate the value of a single pulse in accordance with our canonical functional form. the form is a sigmoid
    rise with an exponential decay that sits on top of a constant offset.
    :param A1: pulse amplitude
    :param k1: sigmoid rise steepness
    :param k2: exponential decay time constant
    :param x1: sigmoid positions
    :param C: constant offset
    :param x: positions at which to evaluate this function
    :return:
    """

    return (logistic(A1, k1, x1, x) * decay(1.0, k2, x1, x))  + C


def pulseAmplitude(A, k1,  k2,  x0):
    """
     This function computes the amplitude of a pulse given its parameters. Note, the return value will be negative
     for pulses where k1/k2 <= 1.
    :param A: Scaling term of the pulse
    :param k1: steepness term of the logistic
    :param k2: the fall time term of the decay
    :param x0: the position of the pulse
    :return: the amplitude of the fitted pulse
    """

    frac = k1/k2
    if frac <= 1.0:
        return -1

    pos = x0 + math.log(frac-1.0)/k1;
    return singlePulse(A, k1, k2, x0, 0.0, pos)

