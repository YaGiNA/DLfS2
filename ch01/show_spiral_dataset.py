import sys
sys.path.append("..")   #In order to import from parent dir.
from dataset import spiral
import matplotlib.pyplot as plt


x, t = spiral.load_data()
print(x, x.shape)
print(t, t.shape)
