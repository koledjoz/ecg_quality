from distutils.core import setup
import setuptools

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
  name = 'ecg_quality',
  packages = setuptools.find_packages(),
  package_data={'ecg_quality': ['models/*.h5']},
  long_description=long_description,
  long_description_content_type='text/markdown',
  version = '0.2.0',
  license='gpl-3.0',
  description = 'Library that classifies quality of ECG signal using deep learning methods',
  author = 'Jozef Koleda',
  author_email = 'koledjoz@cvut.cz',
  url = 'https://github.com/koledjoz/ecg_quality',
  download_url = 'https://github.com/koledjoz/ecg_quality/releases/tag/v0.2.0',
  keywords = ['ECG', 'quality', 'classification', 'deep learning'],
  install_requires=[
          'tensorflow',
          'numpy',
          'neurokit2',
      ]
)