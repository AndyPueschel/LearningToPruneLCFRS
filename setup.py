from setuptools import setup
from ltplcfrs import __version__

setup(name='ltplcfrs',
      version=__version__,
      description='Training of a binary pruning policy for LCFRS',
      url='TODO',
      author='Andy Püschel',
      license='Apache License 2.0',
      packages='ltplcfrs',
      requires=['disco-dop', 'pandas'],
      entry_points={'console_scripts': ['ltplcfrs = ltplcfrs.cli:main']},
      zip_safe=False)
