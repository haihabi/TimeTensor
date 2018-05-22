from setuptools import setup, find_packages


setup(
    name="time_tensor",
    version="0.1",
    author="Hai Victor Habi",
    license="MIT",
    packages=find_packages(exclude=['tests', 'example'])
)

