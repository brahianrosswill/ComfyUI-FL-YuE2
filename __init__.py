from .yue2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .yue2.training.nodes import TRAINING_NODES
from .yue2.training import routes  # noqa: F401 - registers local training endpoints

NODE_CLASS_MAPPINGS.update(TRAINING_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update({name: "FL YuE2 · " + label for name, label in zip(TRAINING_NODES, (
    "Training Models", "Gemini Music Captioner", "Dataset Maker", "Prepare Dataset", "Train Config", "LoRA Trainer", "Load LoRA"))})

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
