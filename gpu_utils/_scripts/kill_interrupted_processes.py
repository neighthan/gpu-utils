from subprocess import run, PIPE
from argparse import ArgumentParser
from .. import get_gpus


def kill_interrupted_processes(sudo: bool = False) -> None:
    """
    Kill processes which `nvidia-smi` doesn't show but which are still using GPUs.

    Interrupting a process can sometimes lead to it not properly releasing GPU memory. This
    function tries to regain that memory by killing any processes that `nvidia-smi` doesn't
    show as running but that you can still see are using the GPUs using `lsof /dev/nvidia{gpu_id}`.
    `kill -9 {pid}` will be called for all such processes.

    :param sudo: if True, use sudo to find and kill interruped processes belong to all users
    """
    sudo = "sudo " if sudo else ""
    gpus = get_gpus()
    n_gpus = len(gpus)
    all_pids = set()
    for gpu_id in range(n_gpus):
        stdout = run(
            (sudo + f"lsof -t /dev/nvidia{gpu_id}").split(" "), stdout=PIPE
        ).stdout
        pids = stdout.decode().split()
        all_pids.update(pids)

    running_pids = [process.pid for gpu in gpus for process in gpu.processes]
    kill_pids = all_pids.difference(running_pids)
    run([*(sudo + "kill -9").split(" "), *kill_pids])


def main():
    parser = ArgumentParser(
        description="""Kill processes which `nvidia-smi` doesn't show but which are still using GPUs.

    Interrupting a process can sometimes lead to it not properly releasing GPU memory. This
    function tries to regain that memory by killing any processes that `nvidia-smi` doesn't
    show as running but that you can still see are using the GPUs using `lsof /dev/nvidia{gpu_id}`.
    `kill -9 {pid}` will be called for all such processes."""
    )
    parser.add_argument(
        "-s",
        "--sudo",
        action="store_true",
        help="if True, use sudo to find and kill interrupted processes belong to all users",
    )

    args = parser.parse_args()
    kill_interrupted_processes(args.sudo)


if __name__ == "__main__":
    main()
