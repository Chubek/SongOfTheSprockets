
from setuptools import setup

setup(
    name="sparrow-voice",
    version="1.0.0",
    description="Voice conversion tool",
    url="https://github.com/OctoShrew/voice-conversion",
    author="Chubak Bidpaa",
    author_email="chubak.bidpaa@octoshrew.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
    ],
    packages=["sparrow_voice"],
    include_package_data=True,
    install_requires=[
        "sprocket-vc", "dearpygui", "numpy", "scipy", "pysptk", "scikit-learn", "pyworld", "h5py", "dtw", "fastdtw", "pyyaml", "docopt"
    ],
    entry_points={"console_scripts": ["sparrow_voice=conv_gui.main"]},
)
