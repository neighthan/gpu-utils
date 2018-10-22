# GPU Utils

A few small functions/scripts for working with GPUs.

## Requirements

* Python 3.6+
* Linux OS (only tested on Ubuntu; I use `subprocess.run` for things like `nvidia-smi`, `kill`, and `lsof`)

## Installation

```
pip install gpu_utils
```

## Usage

```python
from gpu_utils.utils import gpu_init

# sets GPU ids to use nvidia-smi ordering (CUDA_DEVICE_ORDER = PCI_BUS_ID)
# finds the gpu with the most free utilization or memory
# hides all other GPUs so you only use this one (CUDA_VISIBLE_DEVICES = <gpu_id>)
# optionally sets up a `tf.ConfigProto` or `torch.device`
gpu_id = gpu_init(best_gpu_metric="util") # could also use "mem"
```

## Command Line Scripts

`gpu` runs `nvidia-smi` and then `ps u <pid>` on the PIDs so you can see who is running what. I plan to make the output more concise than `nvidia-smi` at some point.

`kill_interrupted_processes` is useful if you interrupt a process using a GPU but find that, even though `nvidia-smi` no longer shows the process, the memory is still being held. It will send `kill -9` to all such processes so you can reclaim your GPU memory.

`tmux_gpu_info.py` just prints a list of the percent utilization of each GPU; you can, e.g., show this in the status bar of `tmux` to keep an eye on your GPUs.
