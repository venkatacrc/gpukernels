import os
import pathlib
import sys

sys.path.insert(0, os.environ.get("GK_ROOT", str(pathlib.Path(__file__).resolve().parents[4])))
from serving.hstu_backend import HSTUServer


class TritonPythonModel(HSTUServer):
    pass
