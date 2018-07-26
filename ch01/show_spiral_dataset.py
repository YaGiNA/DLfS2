import sys
sys.path.append("..")   #In order to import from parent dir.
from dataset import spiral
import matplotlib.pyplot as plt


x, t = spiral.load_data()
print(x, x.shape)
print(t, t.shape)

# plotting data
N = 100
CLS_NUM = 3
makers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, maker=makers[i])
plt.show()
