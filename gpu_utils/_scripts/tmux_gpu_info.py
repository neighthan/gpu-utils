# I use this to show the utilisation of each GPU in the status bar in tmux
# e.g. with this line in ~/.tmux.conf:
# set -g status-right '#[fg=yellow]#(tmux_gpu_info.py)'

from .. import get_gpus


def main():
    gpus = get_gpus()

    # list of util_used for each GPU
    print([round(gpu.util_used, 2) for gpu in gpus])


if __name__ == "__main__":
    main()
