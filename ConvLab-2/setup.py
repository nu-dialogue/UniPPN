'''
setup.py for ConvLab-2
'''
import sys
import os
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class LibTest(TestCommand):

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        ret = os.system("pytest --cov=ConvLab-2 tests/ --cov-report term-missing")
        sys.exit(ret >> 8)

setup(
    name='ConvLab-2',
    version='1.0.0',
    packages=find_packages(exclude=[]),
    license='Apache',
    description='Task-oriented Dialog System Toolkits',
    long_description=open('README.md', encoding='UTF-8').read(),
    long_description_content_type="text/markdown",
    classifiers=[
                'Development Status :: 2 - Pre-Alpha',
                'License :: OSI Approved :: Apache Software License',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Intended Audience :: Science/Research',
                'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        'nltk>=3.4',
        'tqdm>=4.30',
        'checksumdir>=1.1',
        'dataclasses',
        'visdom',
        'Pillow',
        'future',
        # 'torch>=1.2.0,<1.6.0',
        'numpy>=1.15.0',
        'scipy',
        # 'scikit-learn==0.20.3',
        'pytorch-pretrained-bert>=0.6.1',
        # 'transformers>=2.3.0,<3.0.0',
        'tensorboard>=1.14.0',
        # 'tensorboardX==1.7',
        # 'tokenizers==0.8.0',
        # 'overrides==4.1.2',
        # 'allennlp==0.9.0',
        'requests',
        'simplejson',
        # 'spacy==2.1.9',
        'unidecode',
        'jieba',
        'embeddings',
        'quadprog',
        'pyyaml',
        'fuzzywuzzy',
        'python-Levenshtein',
        'json_lines',
        'gtts',
        'pydub'
    ],
    extras_require={
        'develop': [
            "python-coveralls",
            "pytest-dependency",
            "pytest-mock",
            "requests-mock",
            "pytest>=3.6.0",
            "pytest-cov==2.4.0",
            "checksumdir",
            "bs4",
            "lxml",
        ]
    },
    cmdclass={'test': LibTest},
    entry_points={
        'console_scripts': [
            "ConvLab-2-report=convlab2.scripts:report"
        ]
    },
    include_package_data=True,
    url='https://github.com/thu-coai/ConvLab-2',
    author='thu-coai',
    author_email='thu-coai-developer@googlegroups.com',
    python_requires='>=3.5',
    zip_safe=False
)
