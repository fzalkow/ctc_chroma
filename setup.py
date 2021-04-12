from setuptools import setup


with open('README.md', 'r') as stream:
    long_description = stream.read()

setup(name='ctc_chroma',
      version='2.1',
      description='CTC-based chroma feature exractors',
      author='Frank Zalkow',
      author_email='frank.zalkow@audiolabs-erlangen.de',
      url='https://github.com/fzalkow/ctc_chroma',
      long_description=long_description,
      long_description_content_type='text/markdown',
      package_data={'': ['audio/*'], 'ctc_chroma': ['models/*']},
      license='MIT',
      packages=['ctc_chroma'],
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 3',
      ],
      keywords='audio music sound',
      install_requires=[
        'numpy == 1.16.*',
        'scipy == 1.*',
        'matplotlib == 2.0.*',
        'ipython == 7.7.*',
        'jupyter == 1.0.*',
        'tqdm == 4.36.*',
        'numba == 0.48.*',
        'librosa == 0.7.*',
        'tensorflow == 2.*',
        ],
      python_requires='>=3.6')
