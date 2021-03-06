from setuptools import setup, find_packages
from cellseg.version import __version__

setup(name='cellseg',
      version=__version__,
      description='Multiclass Cell Segmentation',
      url="http://www.github.com/Nelson-Gon/cellseg",
      download_url="https://github.com/Nelson-Gon/cytounet/archive/v0.0.0.zip",
      author='Nelson Gonzabato',
      author_email='gonzabato@hotmail.com',
      license='MIT',
      keywords="torch pytorch images image-analysis deep-learning biology",
      packages=find_packages(),
      long_description=open('README.md', encoding="UTF-8").read(),
      python_requires='>=3.6',
      long_description_content_type='text/markdown',
      zip_safe=False)
