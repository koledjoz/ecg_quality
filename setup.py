from distutils.core import setup
import setuptools

setuptools.setup(
  name = 'ecg_quality',
  packages = setuptools.find_packages(),
  package_data={'ecg_quality': ['models/*.h5']},
  version = '0.1.2',
  license='gpl-3.0',
  description = 'Library that classifies quality of ECG signal using deep learning methods',
  author = 'Jozef Koleda',
  author_email = 'koledjoz@cvut.cz',
  url = 'https://github.com/koledjoz/ecg_quality',
  download_url = 'https://github.com/koledjoz/ecg_quality/releases/tag/v0.1.2',
  keywords = ['ECG', 'quality', 'classification', 'deep learning'],
  install_requires=[
          'tensorflow',
          'numpy',
          'neurokit2',
      ]
)