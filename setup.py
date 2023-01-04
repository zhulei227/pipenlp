from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()
setup(
    # Metadata
    name='pipenlp',
    version='0.0.1',
    python_requires='>=3.6',
    author='ZhuLei227',
    url='https://github.com/zhulei227/PipeNLP',
    description='NLP Pipeline Toolkit',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    license='Apache-2.0',

    # Package info
    install_requires=requirements)
