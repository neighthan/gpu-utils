import re
from pathlib import Path
from time import sleep
from invoke import task
from invoke.exceptions import UnexpectedExit


@task
def upload(ctx, test: bool = False, n_download_tries: int = 3):
    """
    Assumptions:
    * the project is structured as <name>/<name'> where <name>
      is the project name and <name'> is the same as <name> except with any "-"
      replaced by "_". `tasks.py` should be stored in <name>.
    * the version is stored in <name>/<name'>/_version.py and that
      this file only contains `__version__ = "<version>"` where <version> is of the
      format major.minor.micro (with optional suffix such as dev0).
    * If you use the test flag, you have at least the following in `~/.pypirc`:
      [testpypi]
      repository: https://test.pypi.org/legacy/
    """

    # TODO - in case version has a suffix besides dev, we should read the whole suffix
    # replace it by dev<dev_num>, do the test, then write back the original suffix
    # again. If the suffix isn't dev, though, we won't know what to set dev_num to
    # so we'll need to run pip to check what the latest version is that has the same
    # major.minor.micro and dev; then we can pull dev_num from that and increment
    # pip install --index-url https://test.pypi.org/simple/ gpu-utils==? then use re
    # to parse out all versions; make a pattern from the regex below so you can use
    # it in both places.

    # TODO - don't upload if test is False and there are unstaged changes to
    # tracked files unless --force is given

    # TODO - add a git tag to the most recent commit that says what the version is
    # https://stackoverflow.com/questions/4404172/how-to-tag-an-older-commit-in-git

    # TODO - put this into a separate repo that you use as a template for python
    # projects

    project_root = "/cluster/nhunt/github/gpu-utils"
    project_name = "gpu-utils"
    sleep_time = 5

    version = {}
    version_path = (
        Path(__file__).parent / project_name.replace("-", "_") / "_version.py"
    )
    exec(version_path.read_text(), version)
    version = version["__version__"]

    if test:
        # add dev if lacking; increment dev number if present
        # this is because test.pypi still won't let you upload the same version
        # multiple times. Doing this automates changing the version for repeat testing

        versions = re.fullmatch(
            r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<micro>\d+)(dev(?P<dev_num>\d+))?",
            version,
        )
        dev_num = versions.groupdict()["dev_num"]
        if dev_num:
            version = version.replace(f"dev{dev_num}", f"dev{int(dev_num) + 1}")
        else:
            version += "dev0"
    else:
        # remove dev if present
        try:
            version = version[: version.index("dev")]
        except ValueError:
            pass

    # write back the modified version
    version_path.write_text(f'__version__ = "{version}"\n')

    cmd = f"""
    cd {project_root}

    rm -rf build
    rm -rf dist

    pip install -U setuptools wheel twine
    python setup.py sdist bdist_wheel
    twine upload {'--repository testpypi' if test else ''} dist/*

    rm -rf build
    rm -rf dist
    """

    ctx.run(cmd)

    if not test:
        return

    for i in range(n_download_tries):
        sleep(sleep_time)
        try:
            result = ctx.run(
                f"""
            cd
            pip install --index-url https://test.pypi.org/simple/ {project_name}=={version}
            """
            )
            break
        except UnexpectedExit:
            continue
