from setuptools import setup

setup(
    name="animal",
    version="0.0.1dev",
    install_requires=[
        "numpy",
        "torch>=1.11",
        "gym",
        "click",
        "kindling @ git+https://github.com/jfpettit/kindling@main",
        "tqdm",
        "tensorboardX"
    ]
)