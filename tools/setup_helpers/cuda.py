import os
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .env import check_env_flag


def find_nvcc():
    proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    return os.path.dirname(out) if len(out) > 0 else None


if check_env_flag('NO_CUDA'):
    WITH_CUDA = False
    CUDA_HOME = None
else:
    CUDA_HOME = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        osname = platform.system()
        if osname == 'Linux':
            cuda_path = find_nvcc()
        else:
            cudart_path = ctypes.util.find_library('cudart')
            cuda_path = os.path.dirname(cudart_path) if cudart_path is not None else None
        CUDA_HOME = os.path.dirname(cuda_path) if cuda_path is not None else None
    WITH_CUDA = CUDA_HOME is not None
