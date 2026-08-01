import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gtrace",
    version="0.2.4",
    author="Yoichi Aso",
    author_email="yoichi.aso@nao.ac.jp",
    description="2D ray tracing package for Gaussian beams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asoy01/gtrace",
    packages=setuptools.find_packages(),
    package_data={'gtrace.draw.viewer': ['*.js', '*.css', '*.html']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ['numpy>=1.5.0', 'scipy>=0.1.0','traits>=4.0.0'],
    # The HTML viewer needs nothing extra; only the notebook widget does.
    extras_require = {'notebook': ['anywidget>=0.9']}
)


