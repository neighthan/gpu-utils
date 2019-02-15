#!/usr/bin/env python

# I use this to show the utilisation of each GPU in the status bar in tmux
# e.g. with this line in ~/.tmux.conf:
# set -g status-right '#[fg=yellow]#(tmux_gpu_info.py)'

from gpu_utils import get_gpus

gpus = get_gpus()

# list of util_used for each GPU
print([round(gpu.util_used, 2) for gpu in gpus])
