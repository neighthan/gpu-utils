import os
import re
from collections import namedtuple
from subprocess import run, PIPE
from typing import Sequence, List, Optional

# mem in MiB, util as % used
GPU = namedtuple("GPU", ["num", "mem_used", "mem_free", "util_used", "util_free"])


def nvidia_smi(all_output: bool = False) -> str:
    """
    :param all_output: whether to return all output from nvidia-smi. If true, `nvidia-smi` (no flags) is run. Otherwise,
        `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv` is run.
        This results in a format that is easier to parse to get memory and utilization information, but
        it doesn't contain all information that `nvidia-smi` does by default.
    :returns: standard output from nvidia-smi
    """

    smi_command = "nvidia-smi"
    if not all_output:
        smi_command += (
            " --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv"
        )
    return run(smi_command.split(" "), stdout=PIPE).stdout.decode()


def get_n_gpus() -> int:
    stdout = run("nvidia-smi -L".split(" "), stdout=PIPE).stdout
    n_gpus = len(stdout.decode().strip().split("\n"))
    return n_gpus


def get_gpus(
    skip_gpus: Sequence[int] = (), smi_info: str = "", keep_all: bool = False
) -> List[GPU]:
    """
    :param skip_gpus: which GPUs not to include in the list
    :param smi_info: info from calling `nvidia_smi()`; if not given, this is generated
    :param keep_all: whether to keep all GPUs in the returned list, even those that don't show utilization in nvidia-smi;
        util_free and util_used will be None for such GPUs if they're kept
    :returns: a list of namedtuple('GPU', ['num', 'mem_used', 'mem_free', 'util_used', 'util_free'])
    """

    if not smi_info:
        smi_info = nvidia_smi()

    gpus = []
    for line in smi_info.strip().split("\n")[1:]:  # 0 has headers
        num, mem_used, mem_total, util_used = line.split(", ")

        num = int(num)
        if num in skip_gpus:
            continue

        mem_used = int(mem_used.split(" ")[0])
        mem_total = int(mem_total.split(" ")[0])
        mem_free = mem_total - mem_used

        try:
            util_used = int(util_used.split(" ")[0])
            util_free = 100 - util_used
        except ValueError:  # utilization not supported
            if not keep_all:
                continue
            util_used = None
            util_free = None

        gpus.append(GPU(num, mem_used, mem_free, util_used, util_free))

    return gpus


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

    return best_gpu.num


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


def get_running_pids(smi_info: Optional[str] = None) -> List[str]:
    """
    :param smi_info: from running `nvidia_smi(all_output=True)`; generated if not given
    """
    smi_info = smi_info or nvidia_smi(all_output=True)
    # just by inspection; PID lines start "|    gpu_id    pid"
    pid_matcher = re.compile(r"\|\s+\d\s+(\d+)")
    pids = re.findall(pid_matcher, smi_info)
    return pids
