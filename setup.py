import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sigmapy",
    version="0.1",
    author="Jacob Downs",
    author_email="jacob.downs@umontana.edu",
    description="A package of algorithms for generating sigma point and \
      weight sets for Bayesian inference problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JacobDowns/sigmapy",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'GPy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
