import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="doubletake",
    version=__version__,
    description="DoubleTake: Geometry Guided Depth Estimation",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": "https://github.com/nianticlabs/doubletake"},
)
