import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="mvsanywhere",
    version=__version__,
    description="MVSAnywhere: Zero Shot Multi-View Stereo",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": "https://github.com/nianticlabs/mvsanywhere"},
)
