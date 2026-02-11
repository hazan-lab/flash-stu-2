"""
MiniSTU: Lightweight Spectral Transform Unit

A simplified implementation of STU focused on the core spectral filtering innovation,
perfect for learning dynamical systems and research applications.
"""

from .stu import MiniSTU
from .filters import get_spectral_filters, get_hankel
from .convolution import convolve, flash_convolve
from .lds import LDS, random_LDS, train_stu_on_lds
from .utils import nearest_power_of_two

__version__ = "1.0.0"

__all__ = [
    "MiniSTU",
    "get_spectral_filters",
    "get_hankel",
    "convolve",
    "flash_convolve",
    "LDS",
    "random_LDS",
    "train_stu_on_lds",
    "nearest_power_of_two",
]

