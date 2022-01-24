
from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

wtuq_require = [
    "numpy",
    "matplotlib",
    "pandas",
    "configobj",
    "bokeh",
    "uncertainpy @ git+https://github.com/DLR-AE/uncertainpy.git",
    "pydmd",
    "geomdl",
]

setup(
    name="wtuq_framework",
    version="0.0.1",
    author="Hendrik Verdonck & Oliver Hach",
    author_email="hendrik.verdonck@dlr.de",
    description="Uncertainpy wrapper for wind turbine specific analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DLR-AE/wtuq",
    packages=['wtuq_framework'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MS Windows",
    ],
    python_requires='>=3.6',
    install_requires=wtuq_require,
)

