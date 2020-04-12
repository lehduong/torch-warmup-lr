import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='torch_warmup_lr',
    version='1.0.0',
    author='Le H. Duong',
    description=('a warpper for pytorch lr_scheduler that support warmup learning rate'),
    license='',
    keywords='learning_rate warmup pytorch',
    packages=find_packages(),
    install_requires=[
        'torch'
    ],
)
