# GPU Utils

A few small functions/scripts for working with GPUs.

## Requirements

* Python 3.6+
* Linux OS for full functionality (only tested on Ubuntu; I use `subprocess.run` for `kill` and `lsof`)
  * Everything except `kill_interrupted_processes` should work on any OS

## Installation

```
pip install gpu-utils
```

The PyPI page is [here][pypi page].

## Usage

```python
from gpu_utils import gpu_init

# sets GPU ids to use nvidia-smi ordering (CUDA_DEVICE_ORDER = PCI_BUS_ID)
# finds the gpu with the most free utilization or memory
# hides all other GPUs so you only use this one (CUDA_VISIBLE_DEVICES = <gpu_id>)
gpu_id = gpu_init(best_gpu_metric="util") # could also use "mem"
```

If you use TensorFlow or PyTorch, `gpu_init` can take care of another couple of steps for you:

```python
# a torch.device for the selected GPU
device = gpu_init(ml_library="torch")
```

```python
import tensorflow as tf
# a tf.ConfigProto to allow soft placement + GPU memory growth
config = gpu_init(ml_library="tensorflow")
session = tf.Session(config=config)
```

## Command Line Scripts

`gpu` is a more concise and prettier version of `nvidia-smi`. It is similar to [`gpustat`][gpustat] but with more control over the color configuration and the ability to show the full processes running on each GPU.

`kill_interrupted_processes` is useful if you interrupt a process using a GPU but find that, even though `nvidia-smi` no longer shows the process, the memory is still being held. It will send `kill -9` to all such processes so you can reclaim your GPU memory.

`tmux_gpu_info.py` just prints a list of the percent utilization of each GPU; you can, e.g., show this in the status bar of `tmux` to keep an eye on your GPUs.

## Acknowledgements
* Using `pynvml` instead of parsing `nvidia-smi` with regular expressions made this library a bit faster and much more robust than my previous regex parsing of `nivida-smi`'s output; thanks to [`gpustat`][gpustat] for showing me this library and some ideas about the output format for the `gpu` script.

[pypi page]: https://pypi.org/project/gpu-utils/
[gpustat]: https://github.com/wookayin/gpustat
