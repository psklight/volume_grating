import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="volume_grating",
    version="0.1.1",
    author="Pisek Kultavewuti",
    author_email="psk.light@gmail.com",
    description="tools to analyze volume gratings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psklight/volume_grating",
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'pandas', 'sympy', 'tqdm', 'matplotlib', 'scipy'],
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License"
    ),
)