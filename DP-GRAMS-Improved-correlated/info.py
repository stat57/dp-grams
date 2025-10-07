import os, platform, psutil

print("OS:", platform.system(), platform.release())
print("CPU:", platform.processor())
print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical processors:", psutil.cpu_count(logical=True))
print("RAM:", round(psutil.virtual_memory().total / 1e9, 2), "GB")

import sys
import platform
import sklearn
import numpy
import matplotlib
import seaborn
import diffprivlib

print("Python version:", sys.version)
print("Platform:", platform.platform())
print("NumPy version:", numpy.__version__)
print("scikit-learn version:", sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", seaborn.__version__)
print("Diffprivlib version:", diffprivlib.__version__)
