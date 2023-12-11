import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="geometryhints",
    version=__version__,
    description="GeometryHints Research",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": "https://gitlab.nianticlabs.com/niantic-ar/research/GeometryHints"},
)