import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import uuid

from filelock import FileLock
import folder_paths
import comfy.model_management as mm
from server import PromptServer

from .data import read_run, write_run, write_json


def output_root():
    return Path(folder_paths.get_output_directory()) / "yue2_training"


def run_worker(request, node_id=None, client_id=None, *, api_key=""):
    if client_id is None and hasattr(PromptServer, "instance"):
        client_id = PromptServer.instance.client_id
    jobs = output_root() / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job = jobs / (uuid.uuid4().hex + ".json")
    write_json(job, request)
    cancel_path = job.with_suffix(".cancel")
    result, error, process = None, None, None
    events = queue.Queue()
    lock = FileLock(str(output_root() / "gpu.lock"))
    with lock.acquire(timeout=0):
        if request["operation"] != "caption":
            mm.unload_all_models()
            mm.soft_empty_cache()
        try:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("worker.py")), str(job)],
                                       cwd=str(Path(__file__).resolve().parents[4]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       stdin=subprocess.PIPE if request["operation"] == "caption" else subprocess.DEVNULL,
                                       text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if request["operation"] == "caption":
                process.stdin.write(json.dumps(api_key) + "\n")
                process.stdin.close()
            def read():
                for line in process.stdout:
                    events.put(line)
                events.put(None)
            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            with job.with_suffix(".log").open("w", encoding="utf-8") as log:
                done = False
                while not done:
                    mm.throw_exception_if_processing_interrupted()
                    try:
                        line = events.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if line is None:
                        done = True
                        continue
                    log.write(line)
                    log.flush()
                    if line.startswith("YUE2_EVENT "):
                        event = json.loads(line[len("YUE2_EVENT "):])
                        if event["type"] == "complete":
                            result = event["result"]
                        elif event["type"] == "error":
                            error = event["message"]
                        if node_id is not None and hasattr(PromptServer, "instance"):
                            operation = {"paired_train": "train", "paired_preview": "preview"}.get(request["operation"], request["operation"])
                            PromptServer.instance.send_sync("fl_yue2.training", {"node": str(node_id), "job": job.stem, "operation": operation, **event}, client_id)
                process.wait()
            if process.returncode or result is None:
                raise RuntimeError(error or f"YuE2 worker failed; see {job.with_suffix('.log')}")
            return result
        except mm.InterruptProcessingException:
            cancel_path.touch()
            if node_id is not None and hasattr(PromptServer, "instance"):
                PromptServer.instance.send_sync("fl_yue2.training", {"node": str(node_id), "type": "error", "message": "Cancelled; the latest saved checkpoint can be resumed."}, client_id)
            raise
        finally:
            if process is not None and process.poll() is None:
                cancel_path.touch()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
            if process is not None:
                process.stdout.close()
            if cancel_path.exists() and request["operation"] in ("train", "paired_train"):
                path = Path(request["run_directory"]) / "run.json"
                if path.exists():
                    run = read_run(path)
                    if run["status"] == "running":
                        run["status"] = "cancelled"
                        write_run(path, run)
            mm.soft_empty_cache()
