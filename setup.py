
from setuptools import setup, find_packages

setup(
    name='xor_gate_nn',
    version='0.1',
    description='A brief description of my package',
    long_description=open('README.md').read(),
    author='Pratap2018',
    author_email='pratapmridha@gmail.com',
    url='https://github.com/Pratap2018/XorNet',
    packages=find_packages(),
    install_requires=[
        'torch',    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
