from .blocks import *
from .resnet_encdec import Encoder, Decoder, EncoderDecoder
from .residual_mlp import ResidualMLP
from .convnext_ae import ConvNextEncoder, ConvNextDecoder, ConvNextAutoencoder
from .GNWrapper import GNWrapper

# Auto-register all classes manually imported
import inspect
import sys

_backbone_registry = {}

for name, obj in list(globals().items()):
    if inspect.isclass(obj):
        _backbone_registry[name.lower()] = obj

def get_backbone(name: str):
    return _backbone_registry.get(name.lower())