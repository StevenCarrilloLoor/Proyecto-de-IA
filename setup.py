"""
Setup para el proyecto de Análisis de Baloncesto
"""

from setuptools import setup, find_packages
from pathlib import Path

# Leer README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Leer requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="basketball-analysis",
    version="1.0.0",
    author="Steven Carrillo",
    author_email="tu_email@example.com",
    description="Sistema de Detección y Análisis de Jugadas en Baloncesto mediante Visión por Computador",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StevenCarrilloLoor/Proyecto-de-IA",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'basketball-analyze=main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
