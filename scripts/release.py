import os
import sys
from pathlib import Path
from subprocess import getoutput

PKG_NAME = "codenav"


def make_package(verbose=False):
    """Prepares sdist for codenav."""

    orig_dir = os.getcwd()

    base_dir = os.path.dirname(os.path.abspath(os.path.dirname(Path(__file__))))
    os.chdir(base_dir)

    with open(".VERSION", "r") as f:
        __version__ = f.readline().strip()

    # generate sdist via setuptools
    output = getoutput(f"{sys.executable} setup.py sdist")
    if verbose:
        print(output)

    os.chdir(os.path.join(base_dir, "dist"))

    # uncompress the tar.gz sdist
    output = getoutput(f"tar zxvf {PKG_NAME}-{__version__}.tar.gz")
    if verbose:
        print(output)

    # create new source file with version
    getoutput(
        f"printf '__version__ = \"{__version__}\"\n' >> {PKG_NAME}-{__version__}/{PKG_NAME}/_version.py"
    )
    # include it in sources
    getoutput(
        f'printf "\ncodenav/_version.py" >> {PKG_NAME}-{__version__}/{PKG_NAME}.egg-info/SOURCES.txt'
    )

    # recompress tar.gz
    output = getoutput(
        f"tar zcvf {PKG_NAME}-{__version__}.tar.gz {PKG_NAME}-{__version__}/"
    )
    if verbose:
        print(output)

    # remove temporary directory
    output = getoutput(f"rm -r {PKG_NAME}-{__version__}")
    if verbose:
        print(output)

    os.chdir(orig_dir)


if __name__ == "__main__":
    make_package(verbose=False)
