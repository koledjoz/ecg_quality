from distutils.core import setup
setup(
  name = 'ecq_quality',         # How you named your package folder (MyLib)
  packages = ['ecq_quality'],   # Chose the same as "name"
  version = '0.1.0',      # Start with a small number and increase it with every change you make
  license='gpl-3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library that clasifies quality of ECG signal using deep learning methods',   # Give a short description about your library
  author = 'Jozef Koleda',                   # Type in your name
  author_email = 'koledjoz@cvut.cz',      # Type in your E-Mail
  url = 'https://github.com/koledjoz/ecq_quality',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['ECG', 'quality', 'classification', 'deep learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'tensorflow',
          'numpy',
          'neurokit2',
      ]
)