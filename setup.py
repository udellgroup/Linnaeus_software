import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="linnaeus",
  version="0.0.1",
  author="Shipu Zhao, Laurent Lessard, Madeleine Udell",
  author_email="",
  description="A Python package for checking equivalence and relations between iterative algorithms",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/QCGroup/linnaeus",
  packages=setuptools.find_packages(),
  install_requires=['sympy >= 1.5.1',
                    'numpy >= 1.16',
                    'scipy >= 1.2.1'],
  test_suite='nose2.collector.collector',
  tests_require=['nose2'],
  classifiers=[
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
  ],
)