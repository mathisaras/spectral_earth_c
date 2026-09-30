"""
Thin wrapper around DofaBasePatch16/DofaLargePatch16 that automatically
calls load_encoder_weights() after construction, since the base class
requires this to be called explicitly and nothing in the training
pipeline does so today. This file does not modify dofa_encoder.py at all.
"""
import logging
from .dofa_encoder import DofaBasePatch16, DofaLargePatch16


def DofaBasePatch16AutoLoad(**kwargs):
    model = DofaBasePatch16(**kwargs)
    if kwargs.get("encoder_weights"):
        model.load_encoder_weights(logging.getLogger("dofa_autoload"))
    return model


def DofaLargePatch16AutoLoad(**kwargs):
    model = DofaLargePatch16(**kwargs)
    if kwargs.get("encoder_weights"):
        model.load_encoder_weights(logging.getLogger("dofa_autoload"))
    return model
