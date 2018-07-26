import sys
sys.path.append("..")   #In order to import from parent dir.
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet


# Phase1: Settings of hyper parameter
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# Phase2: vars of training
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # Phase3: Shuffle data
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters * batch_size:(iters + 1) * batch_size]
        batch_t t[iters*batch_size:(iters+1)*batch_size]

        # Phase4: Get loss, update parameter
        loss = model.forward(batch_x, batch_t)
        model.backward
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # Phase5: output info of train
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                "| epoch %d | iter %d / %d | loss %.2f" % (
                    epoch + 1, iters + 1, max_iters, avg_loss
                )
            )
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
