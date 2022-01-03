# coding=utf-8
from config import settings

from pathlib import Path
import subprocess
import os

"""
usage: torch-model-archiver [-h] --model-name MODEL_NAME  --version MODEL_VERSION_NUMBER
                      --model-file MODEL_FILE_PATH --serialized-file MODEL_SERIALIZED_PATH
                      --handler HANDLER [--runtime {python,python2,python3}]
                      [--export-path EXPORT_PATH] [-f] [--requirements-file]
"""
root = Path(os.path.dirname(os.path.realpath('__file__')))

subprocess.run(["torch-model-archiver",
                '--model-name', settings.torch_serve.model_name,
                '--version', str(settings.torch_serve.version),
                '--model-file', str((root / settings.torch_serve.model_file_path).resolve()),
                '--export-path', str((root / settings.torch_serve.export_path).resolve()),
                '--handler', str((root / settings.torch_serve.handler_path).resolve()),
                '--force'])
