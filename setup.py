import setuptools
from pathlib import Path


long_description = (Path(__file__).parent / "README.md").read_text()

version = {}
exec((Path(__file__).parent / "gpu_utils/_version.py").read_text(), version)

setuptools.setup(
    name="gpu_utils",
    version=version["__version__"],
    author="Nathan Hunt",
    author_email="neighthan.hunt@gmail.com",
    description="Utility functions for working with GPUs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/neighthan/gpu-utils",
    packages=setuptools.find_packages(),
    scripts=[
        "scripts/gpu",
        "scripts/tmux_gpu_info.py",
        "scripts/kill_interrupted_processes",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)

config_folder = Path("~/.config/gpu_utils").expanduser()
config_dest = config_folder / "gpu_printing_config.py"
config_src = Path(__file__).resolve().parent / "gpu_printing_config.py"

if not config_dest.exists():
    config_folder.mkdir(parents=True, exist_ok=True)
    config_dest.write_text(config_src.read_text())

(config_folder / "__init__.py").touch()
