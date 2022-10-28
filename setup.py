from importlib.machinery import SourceFileLoader
from os.path import exists

from setuptools import find_packages, setup

try:
    version = (
        SourceFileLoader(
            "dual_cyclegan.version",
            "dual_cyclegan/version.py",
        )
        .load_module()
        .version
    )
except FileNotFoundError as e:
    print(str(e))
    version = "0.0.1"
    print("Somethinig is not working :(. Manually set version to {}".format(version))

packages = find_packages()

if exists("README.md"):
    with open("README.md", "r") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""

setup(
    name="dual_cyclegan",
    version=version,
    description="Neural super resolution",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    packages=packages,
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch >= 1.7.0",
        "hydra-core >= 1.1.0",
        "hydra_colorlog",
        "tqdm",
        "soundfile",
        "librosa >= 0.8.0",
        "numba >= 0.50",
        "torchaudio >= 0.8.1",
        "tensorboard",
    ],
    extras_require={
        "train": [],
        "test": [],
    },
    entry_points={
        "console_scripts": [
            "dual-cyclegan-train = dual_cyclegan.bin.train:main",
            "dual-cyclegan-infer = dual_cyclegan.bin.infer:main",
        ],
    },
)
