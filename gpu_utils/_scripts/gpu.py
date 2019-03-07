from argparse import ArgumentParser
from ..utils import get_gpus, get_gpu_string


def print_gpu_info() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-mw",
        "--max_cmd_width",
        type=int,
        default=125,
        help="Width at which to truncate commands.",
    )
    parser.add_argument(
        "-hc",
        "--hide_cmd",
        action="store_true",
        help="Flag to hide commands running on each gpu.",
    )
    parser.add_argument(
        "-nc",
        "--no_color",
        action="store_true",
        help="Flag to remove color from output.",
    )

    args = parser.parse_args()
    gpus = get_gpus(include_processes=not args.hide_cmd)
    print(get_gpu_string(gpus, args.max_cmd_width, args.hide_cmd, args.no_color))


if __name__ == "__main__":
    print_gpu_info()
