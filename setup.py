import setuptools

# The encoding has to be explicit. open() otherwise decodes with the
# locale's preferred encoding, so on a Japanese Windows this reads
# README.md as cp932 and dies on the first non-ASCII character - which
# means the package cannot be built or installed from source there at
# all, while working fine everywhere UTF-8 is the default.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gtrace",
    version="0.6.0",
    author="Yoichi Aso",
    author_email="asoy01@gmail.com",
    description="2D ray tracing package for Gaussian beams",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asoy01/gtrace",
    # PyPI has no changelog field of its own - the project page shows the
    # long description and nothing else - so the way to point at one is a
    # project URL, which PyPI renders as a link in the sidebar.
    project_urls={
        "Documentation": "https://gtrace.readthedocs.io/",
        "Source": "https://github.com/asoy01/gtrace",
        "Changelog": "https://github.com/asoy01/gtrace/blob/master/CHANGELOG.md",
        "Releases": "https://github.com/asoy01/gtrace/releases",
    },
    packages=setuptools.find_packages(),
    package_data={'gtrace.draw.viewer': ['*.js', '*.css', '*.html']},
    # LICENSE grants the two BSD conditions with no non-endorsement
    # clause, which is BSD-2-Clause rather than BSD-3-Clause. The SPDX
    # identifier replaces the "License :: OSI Approved :: BSD License"
    # classifier, which setuptools deprecates and which did not say
    # which BSD. Note that this lands in the legacy License field, not
    # in License-Expression: PEP 639 metadata would mean moving the
    # whole project description into pyproject.toml.
    license="BSD-2-Clause",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # numpy 2.x, which the package is developed and tested against, does
    # not support anything older. Claiming 3.6 only makes pip on an old
    # interpreter fail later and less clearly.
    python_requires='>=3.9',
    install_requires = ['numpy>=1.5.0', 'scipy>=0.1.0','traits>=4.0.0'],
    # The HTML viewer needs nothing extra; only the notebook widget does.
    extras_require = {'notebook': ['anywidget>=0.9']}
)


