"""One queued operation, one owned process; no ComfyUI server in this process."""
import importlib
import json
import os
from pathlib import Path
import sys
import traceback
import types


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    request_path = Path(sys.argv[1])
    sys.argv = [sys.argv[0]]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request["operation"] == "caption":
        request["api_key"] = json.loads(sys.stdin.readline())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root.parents[2]))
    package = types.ModuleType("fl_yue2_worker")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    cancel_path = request_path.with_suffix(".cancel")

    def cancelled():
        if cancel_path.exists():
            raise InterruptedError("YuE2 operation cancelled")

    def emit(value):
        print("YUE2_EVENT " + json.dumps(value, allow_nan=False), flush=True)

    functions = {"prepare": ("training.prepare", "prepare"), "train": ("training.trainer", "train"), "caption": ("training.captioning", "caption"),
                 "preview": ("training.preview", "preview"), "paired_prepare": ("training.paired_prepare", "prepare_pairs"),
                 "paired_train": ("training.paired_train", "train_pairs"), "paired_preview": ("training.paired_infer", "preview_pairs"),
                 "paired_infer": ("training.paired_infer", "infer")}
    try:
        module, function = functions[request["operation"]]
        result = getattr(importlib.import_module(f"{package.__name__}.{module}"), function)(request, emit, cancelled)
        emit({"type": "complete", "result": result})
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "error", "message": str(error), "cancelled": isinstance(error, InterruptedError)})
        sys.exit(1)


if __name__ == "__main__":
    main()
