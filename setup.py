from distutils.core import setup
setup(
  name = 'ecq_quality',
  packages = ['ecq_quality'],
  version = '0.1.0',
  license='gpl-3.0',
  description = 'Library that clasifies quality of ECG signal using deep learning methods',
  author = 'Jozef Koleda',
  author_email = 'koledjoz@cvut.cz',
  url = 'https://github.com/koledjoz/ecq_quality',
  download_url = 'https://github.com/koledjoz/ecq_quality/releases/tag/v0.1.0',
  keywords = ['ECG', 'quality', 'classification', 'deep learning'],
  install_requires=[
          'tensorflow',
          'numpy',
          'neurokit2',
      ]
)