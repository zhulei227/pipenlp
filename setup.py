from setuptools import find_packages, setup
with open('requirements.txt') as f:
    requirements = f.read()
setup(
    name='pipenlp',
    version='0.0.1',
    python_requires='>=3.6',
    author='zhulei227',
    url='https://github.com/zhulei227/pipenlp',
    description='NLP Pipeline Toolkit',
    packages=find_packages(),
    license='Apache-2.0',
    install_requires=requirements)
