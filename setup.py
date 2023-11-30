import setuptools

__version__ = "0.1.0"

setuptools.setup(
    name="simplerecon",
    version=__version__,
    description="SimpleRecon Research",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    project_urls={"Source": "https://gitlab.nianticlabs.com/niantic-ar/research/SimpleRecon"},
)
