import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gtrace",
    version="0.2.0",
    author="Yoichi Aso",
    author_email="yoichi.aso@nao.ac.jp",
    description="2D ray tracing package for Gaussian beams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['numpy>=1.5.0', 'scipy>=0.1.0','traits>=4.0.0']
)


