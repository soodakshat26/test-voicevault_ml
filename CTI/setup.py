from setuptools import setup, find_packages

setup(
    name="voicevault",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "librosa",
        "pywavelets",
        "scikit-learn",
        "matplotlib",
        "sounddevice",
        "soundfile",
        "fastdtw",
        "transformers",
        "torch",
    ],
)
