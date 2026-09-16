import importlib
import sys
import types
from pathlib import Path


def load_transcriber(assets, device="cuda"):
    name = "fl_yue2_sheetsage_compat"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(assets["compat"]))]
    sys.modules[name] = package
    model = types.ModuleType(name + ".model")
    model.ABC_END, model.ABC_START, model.CODEC_SIZE = 151848, 151847, 32768
    model.EOD, model.MUSIC_START = 151643, 151851
    model.INSTRUCTIONS = {}
    sys.modules[name + ".model"] = model
    if "toolkit.basic" not in sys.modules:
        toolkit = types.ModuleType("toolkit")
        toolkit.__path__ = []
        basic = types.ModuleType("toolkit.basic")
        basic.UnusableFileError = ValueError
        sys.modules["toolkit"] = toolkit
        sys.modules["toolkit.basic"] = basic
    module = importlib.import_module(name + ".tokenizer")
    return module.SheetSage2Transcriber(assets["weights"], assets["source"]).to(device)
