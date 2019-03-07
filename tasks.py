"""
Assumptions:
* the project is structured as <name>/<name'> where <name>
  is the project name and <name'> is the same as <name> except with any "-"
  replaced by "_". `tasks.py` should be stored in <name>.
  If you cloned this from gh:neighthan/cookiecutter-pytemplate, the structure is right.
"""

import re
import shutil
import sys
import json
import toml
import invoke
from pathlib import Path
from time import sleep
from typing import Sequence
from urllib.request import urlopen
from urllib.parse import quote
from invoke.exceptions import UnexpectedExit

_version_pattern = re.compile(
    r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)(\.(?P<suffix>[A-z0-9]+))?"
)
_index_url = "--index-url https://test.pypi.org/simple"
_extra_url = "--extra-index-url https://pypi.org/simple"  # for dependencies


@invoke.task
def clean(ctx) -> None:
    """
    Remove all temporary files and directories.

    Files / directories removed are:
      * __pycache__ directories
      * .coverage files
      * build and dist directories
    """

    root_dir = Path(__file__).parent

    rm_file_patterns = [".coverage"]

    for file_pattern in rm_file_patterns:
        for rm_file in root_dir.rglob(file_pattern):
            rm_file.unlink()

    rm_dir_patterns = ["__pycache__", "build", "dist"]

    for dir_pattern in rm_dir_patterns:
        for rm_dir in root_dir.rglob(dir_pattern):
            shutil.rmtree(str(rm_dir))


@invoke.task(clean, post=[clean])
def publish(
    ctx, test: bool = False, install: bool = False, n_download_tries: int = 3
) -> None:
    """
    Publish the project to pypi / testpypi.

    If you use the test flag, you have at least the following in `~/.pypirc`:
      [testpypi]
      repository: https://test.pypi.org/legacy/

    :param ctx: invoke context
    :param test: whether to publish to normal or test pypi. If publishing to testpypi,
      .dev<dev_num> is added to the version where <dev_num> is one larger than the
      highest dev version published. This is because testpypi won't let you publish the
      same version multiple times; doing this automates changing the version for repeat
      publishing + testing.
      Additionally, the micro/patch version is incremented because it's assumed that
      it's a dev version of the _next_ release.
      WARNING - don't publish multiple times too quickly. If so, the next dev num
      can't be pulled from testpypi because it won't have updated yet.
    :param install: whether to install the project from test pypi.
      Only used if `test` is true. This is better than running `invoke install`
      separately because it will try multiple times to get the newly uploaded version
      (it usually takes a couple of tries).
    :param n_download_tries: how many times to attempt to install the project.
      After each attempt there is a 5 second sleep period.
    """

    project_name = _get_from_pyproject(["tool", "poetry", "name"])
    project_root = str(Path(__file__).parent.resolve())
    sleep_time = 5

    if test:
        pyproject_path = Path(__file__).parent / "pyproject.toml"
        original_pyproject_str = pyproject_path.read_text()
        pyproject = toml.loads(original_pyproject_str)
        original_version = pyproject["tool"]["poetry"]["version"]

        version = re.fullmatch(_version_pattern, original_version)
        groups = version.groupdict()
        major, minor, micro = groups["major"], groups["minor"], groups["micro"]
        version = f"{major}.{minor}.{int(micro) + 1}"
        dev_num = _get_next_dev_num(project_name, version)
        version += f".dev{dev_num}"

        # write back the modified version
        pyproject["tool"]["poetry"]["version"] = version
        pyproject_path.write_text(toml.dumps(pyproject))

    try:
        cmd = f"""
        cd "{project_root}"

        poetry build
        twine upload {'--repository testpypi' if test else ''} dist/*
        """

        ctx.run(cmd)
    finally:
        if test:
            pyproject_path.write_text(original_pyproject_str)

    if not test or not install:
        return

    for i in range(n_download_tries):
        sleep(sleep_time)
        try:
            result = ctx.run(
                f"pip install {_index_url} {_extra_url} {project_name}=={version}"
            )
            break
        except UnexpectedExit:
            continue


@invoke.task
def update_tasks(ctx) -> None:
    """
    Update the tasks file to the newest version on GitHub.

    :param ctx: invoke context
    """

    tasks_path = Path(__file__).resolve()
    github_url = "https://raw.githubusercontent.com/neighthan/cookiecutter-pytemplate/"
    github_url += quote("master/{{cookiecutter.project_name}}/tasks.py")

    with urlopen(github_url) as new_tasks_file:
        tasks_path.write_text(new_tasks_file.read().decode())


@invoke.task
def install(ctx, version: str = "", test: bool = False) -> None:
    """
    Install the latest version of the current project.

    :param ctx: `invoke` context
    :param test: whether to install from test pypi;
      if so, `--pre` is used to allow dev versions
    """

    project_name = _get_from_pyproject(["tool", "poetry", "name"])
    cmd = "pip install -U {} " + project_name + f"=={version}" if version else ""
    cmd = cmd.format(" ".join([_index_url, _extra_url, "--pre"]) if test else "")
    ctx.run(cmd)


@invoke.task(clean, allow_unknown=True)
def test(ctx) -> None:
    pytest_args = sys.argv[sys.argv.index("test") + 1 :]
    for i, arg in enumerate(pytest_args):
        if " " in arg:
            pytest_args[i] = f'"{arg}"'
    cmd = "poetry run pytest " + " ".join(pytest_args)
    ctx.run(cmd, pty=True)


@invoke.task
def install_jupyter_kernel(ctx, name: str, install_prefix: str = "") -> None:
    """
    :param name: name for the kernel
    :param install_prefix: default = ~/.local
    """
    install_prefix = Path.home() / ".local"
    cmd = f'poetry run python -m ipykernel install --prefix={install_prefix} --name "{name}"'
    ctx.run(cmd)


def _get_next_dev_num(project_name: str, current_version: str) -> int:
    """
    Get 1 + the number of the latest dev version matching `current_version` w/o suffix.

    To determine next dev_num we run pip and find the latest version that has the same
    major.minor.micro as `current_version` and dev in the suffix, then we increment.
    """

    cmd = f"pip install {_index_url} {project_name}==?"
    result = invoke.run(cmd, warn=True, hide=True)

    current_version = re.fullmatch(_version_pattern, current_version)
    current_version_groups = current_version.groupdict()

    dev_num = 0
    # reverse so that we hit the latest version first
    for published_version in list(re.finditer(_version_pattern, result.stderr))[::-1]:
        groups = published_version.groupdict()
        if (
            groups["major"] == current_version_groups["major"]
            and groups["minor"] == current_version_groups["minor"]
            and groups["micro"] == current_version_groups["micro"]
            and groups["suffix"]
            and groups["suffix"].startswith("dev")
        ):
            dev_num = int(groups["suffix"].replace("dev", "")) + 1
            break
    return dev_num


def _get_from_pyproject(keys: Sequence[str]):
    pyproject = Path(__file__).parent / "pyproject.toml"
    pyproject = toml.loads(pyproject.read_text())
    ret = pyproject
    for key in keys:
        ret = ret[key]
    return ret
