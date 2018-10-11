import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

N = 2   # Size of minibatch
H = 3   # Number of dimension of hidden vec
T = 20  # Length of time data

dh = np.ones((N, H))
np.random.seed(3)   # Set seed of random number due to reproducibility
# Wh = np.random.randn(H, H)
Wh = np.random.randn(H, H) * 0.5


norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# グラフの描画
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xlabel('time step')
plt.ylabel('norm')

plt.savefig('graph.png')
