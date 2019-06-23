from setuptools import setup, find_packages

setup(
    name="time_tensor",
    version="0.1",
    author="Hai Victor Habi",
    license="MIT",
    packages=find_packages(exclude=['tests', 'example'])
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timetensor",
    version="0.1.0",
    author="Hai Victor Habi",
    author_email="victoropensource@gmail.com",
    description="TimeTensor package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haihabi/TimeTensor",
    packages=setuptools.find_packages(exclude=['tests']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)