from setuptools import setup, find_packages

setup(
    name="yolov5",
    version="7.0",
    packages=find_packages(include=['yolov5*']),
    package_dir={'': '.'},
    include_package_data=True,
)