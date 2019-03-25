import os
import psutil
import sys
import pynvml as nv  # nvidia-ml-py3
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, List, Optional, Dict, Union
from colored import fg, bg, attr, names  # https://gitlab.com/dslackw/colored
from .gpu_printing_config import (
    base_format,
    sep,
    colors,
    process_base_format,
    process_sep,
    process_colors,
)


_Process = namedtuple("Process", ["user", "command", "gpu_mem_used", "pid"])


class _GPU:
    """
    Memory is in MiB, utilization is percent used/free.
    """

    def __init__(
        self,
        idx: int,
        mem_used: int,
        mem_total: int,
        util_used: int,
        processes: Optional[List[_Process]] = None,
    ):
        self.idx = idx
        self.mem_used = mem_used
        self.mem_free = mem_total - mem_used
        self.mem_total = mem_total
        self.util_used = util_used
        self.util_free = 100 - util_used
        self.processes = processes if processes is not None else []

    def __repr__(self) -> str:
        repr_attrs = ["idx", "mem_used", "mem_total", "processes"]
        attr_str = ", ".join([f"{attr}={getattr(self, attr)}" for attr in repr_attrs])
        return f"GPU({attr_str})"


class _GPUList(list):
    """List wrapper that does pretty-printing."""

    def __str__(
        self, max_cmd_width: int = 125, hide_cmd: bool = False, no_color: bool = False
    ):
        return get_gpu_string(self, max_cmd_width, hide_cmd, no_color)


# define colors (including attributes) that can be used in format strings
# for pretty-printing GPU info

_attrs = [
    "bold",
    "dim",
    "underlined",
    "blink",
    "reverse",
    "hidden",
    "reset",
    "res_bold",
    "res_dim",
    "res_underlined",
    "res_blink",
    "res_reverse",
    "res_hidden",
]
_fg_colors = {f"fg('{color.lower()}')": fg(color.lower()) for color in names}
_bg_colors = {f"bg('{color.lower()}')": bg(color.lower()) for color in names}
_attrs = {f"attr('{name}')": attr(name) for name in _attrs}
_colors = {**_fg_colors, **_bg_colors, **_attrs}


@contextmanager
def _nvml():
    """Enter a context manager that will init and shutdown nvml."""
    # Copyright (c) 2018 Bohumír Zámečník, Rossum Ltd., MIT license
    # from https://github.com/rossumai/nvgpu/blob/a66dda5ae816a6a8936645fe0520cb4dc6354137/nvgpu/nvml.py#L5
    # Modifications copyright 2019, Nathan Hunt, MIT license

    nv.nvmlInit()
    yield
    nv.nvmlShutdown()


def _try_except_nv_error(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except nv.NVMLError:
        return default


def _to_mb(mem_in_bytes: int) -> int:
    bytes_in_mb = 1024 * 1024
    return int(mem_in_bytes / bytes_in_mb)


def _get_processes(handle: nv.c_nvmlDevice_t) -> List[Dict[str, Union[str, int]]]:
    nv_processes = []
    nv_processes += _try_except_nv_error(
        nv.nvmlDeviceGetComputeRunningProcesses, [], handle
    )
    nv_processes += _try_except_nv_error(
        nv.nvmlDeviceGetGraphicsRunningProcesses, [], handle
    )

    processes = []
    for nv_process in nv_processes:
        try:
            ps_process = psutil.Process(pid=nv_process.pid)
            process = _Process(
                ps_process.username(),
                " ".join(ps_process.cmdline() or ""),
                _to_mb(nv_process.usedGpuMemory),
                nv_process.pid,
            )
            processes.append(process)
        except psutil.NoSuchProcess:
            pass
    return processes


def get_gpus(include_processes: bool = False) -> List[_GPU]:
    """
    Get a list of the GPUs on this machine.

    Any GPUs that don't support querying utilization will have
    util_used == util_free == -1.

    :param include_processes: whether to include a list of the
      processes running on each GPU; this takes more time.
    :returns: a list of the GPUs
    """
    gpus = _GPUList()

    with _nvml():
        for i in range(nv.nvmlDeviceGetCount()):
            handle = nv.nvmlDeviceGetHandleByIndex(i)
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            mem_used = _to_mb(memory.used)
            mem_free = _to_mb(memory.free)
            mem_total = mem_used + mem_free

            try:
                util = nv.nvmlDeviceGetUtilizationRates(handle)
                util_used = util.gpu
            except nv.NVMLError:
                util_used = float("nan")

            processes = _get_processes(handle) if include_processes else []
            gpus.append(_GPU(i, mem_used, mem_total, util_used, processes))
    return gpus


def get_gpus_from_info_string(info_string: str) -> List[_GPU]:
    """
    Get a list of GPUs from output from nvidia-smi.

    :param info_string: the output from running
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
    """
    gpus = _GPUList()
    for line in info_string.strip().replace("MiB", "").replace("%", "").split("\n")[1:]:
        gpus.append(_GPU(*map(int, line.split(", "))))


def get_best_gpu(metric: str = "util") -> int:
    """
    :param metric: one of {util, mem}; "best" means having the largest amount of the desired resource
    :returns: id of the best GPU
    """

    gpus = get_gpus()

    if metric == "util":
        best_gpu = max(gpus, key=lambda gpu: gpu.util_free)
    else:
        assert metric == "mem"
        best_gpu = max(gpus, key=lambda gpu: gpu.mem_free)

    return best_gpu.idx


def gpu_init(
    gpu_id: Optional[int] = None,
    best_gpu_metric: str = "util",
    ml_library: str = "",
    verbose: bool = False,
):
    """
    Set up environment variables CUDA_DEVICE_ORDER and CUDA_VISIBLE_DEVICES.

    If `ml_library` is specified, additional library-specific setup is done.

    :param gpu_id: the PCI_BUS_ID of the GPU to use (the id shown when you run `nvidia-smi`)
      if `None`, the "best" GPU is chosen
    :param best_gpu_metric: one of {util, mem}; which metric to maximize when choosing the best GPU to use
    :param ml_library: one of {torch, tensorflow}; additional setup specific to this library will be done.
      torch: create a `device` using the appropriate GPU
      tensorflow: create a `ConfigProto` that allows soft placement + GPU memory growth
    :param verbose: whether to print the id of the chosen GPU (or that no GPU was found)
    :returns: the id of the GPU chosen if `ml_library == ""`,
      the `torch.device` if `ml_library == "torch"`, or
      the `tf.ConfigProto` if `ml_library == "tensorflow"`
      If no GPUs are found, the id will be `None` but a usable `device` or `ConfigProto` will still be
      returned if one should be. This function should thus be safe to use in code that runs on both GPU-
      equipped and CPU-only machines.
    """
    try:
        gpu_id = gpu_id or get_best_gpu(best_gpu_metric)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if verbose:
            print(f"Running on GPU {gpu_id}.")
    except ValueError:  # no GPUs found
        gpu_id = None
        if verbose:
            print("No GPUs found!")

    if ml_library == "":
        pass
    elif ml_library == "torch":
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    elif ml_library == "tensorflow":
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        return config
    else:
        raise NotImplementedError(f"Support for {ml_library} is not implemented.")

    return gpu_id


def get_gpu_string(
    gpus: Sequence[_GPU],
    max_cmd_width: int = 125,
    hide_cmd: bool = False,
    no_color: bool = False,
) -> str:
    gpu_string = ""

    # combine base format and color commands into the fstrings

    if no_color:
        fstring = sep.join(base_format)
        process_fstring = process_sep.join(process_base_format)
    else:
        fstring = sep.join(
            [
                colors[i] + base_format[i] + "{attr('reset')}"
                for i in range(len(base_format))
            ]
        )

        process_fstring = process_sep.join(
            [
                process_colors[i] + process_base_format[i] + "{attr('reset')}"
                for i in range(len(process_base_format))
            ]
        )

    process_fstring = process_fstring.replace("<max_cmd_width>", str(max_cmd_width))

    for gpu in gpus:
        gpu_string += fstring.format(**vars(gpu), **_colors)
        gpu_string += os.linesep

        if hide_cmd or not gpu.processes:
            continue

        for i, process in enumerate(gpu.processes):
            gpu_string += " └─ " if i == len(gpu.processes) - 1 else " ├─ "
            gpu_string += process_fstring.format(
                **process._asdict(), **_colors, max_cmd_width=max_cmd_width
            )
            gpu_string += os.linesep
    return gpu_string.strip()
