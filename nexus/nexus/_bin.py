from __future__ import annotations

import subprocess
import sys
from os import PathLike
from pathlib import Path

bin_dir = Path(__file__).parent/"bin"
#mth
def run(script_name) -> None:
    script_path = bin_dir/script_name
    result = subprocess.run([script_path] + sys.argv[1:])
    sys.exit(result.returncode)
#end def run

def eshdf() -> None:
    run("eshdf")
#end def eshdf

def nxs_redo() -> None:
    run("nxs-redo")
#end def nxs_redo

def nxs_sim() -> None:
    run("nxs-sim")
#end def nxs_sim

def nxs_test() -> None:
    run("nxs-test")
#end def nxs_test

def qdens() -> None:
    run("qdens")
#end def qdens

def qdens_radial() -> None:
    run("qdens-radial")
#end def qdens_radial

def qmca() -> None:
    run("qmca")
#end def qmca

def qmc_fit() -> None:
    run("qmc-fit")
#end def qmc_fit
