from setuptools import setup, find_packages

setup(
    name='abm_bee_foraging',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'mesa',
        'numpy',
        'pandas',
    ],
    author='Valentin Rexer',
    entry_points={
        'console_scripts': [
            'abm_bee_foraging = abm_bee_foraging.__main__:main',
        ],
    },
)