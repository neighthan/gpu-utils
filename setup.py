import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="gpu_utils",
    version="0.1",
    author="Nathan Hunt",
    author_email="neighthan.hunt@gmail.com",
    description="Utility functions for working with GPUs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neighthan/gpu-utils",
    packages=setuptools.find_packages(),
    scripts=["scripts/gpu", "scripts/tmux_gpu_info.py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
