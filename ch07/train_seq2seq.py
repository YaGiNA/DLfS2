import sys
sys.path.append('..')
from common.np import np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# Reading dataset
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
x_train_rev, x_test_rev = x_train[:, ::-1], x_test[:, ::-1]
char_to_id, id_to_char = sequence.get_vocab()

# Setting hyperparameters
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# Make Model & Optimizer & Trainer

def train_eval(x_train, x_test, is_peeky):
    if is_peeky:
        model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
    else:
        model = Seq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    acc_list = []
    for epoch in range(max_epoch):
        trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
        correct_num = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num += eval_seq2seq(model, question, correct, id_to_char, verbose)
        acc = float(correct_num) / len(x_test)
        acc_list.append(acc)
        print('val acc %.3f%%' % (acc * 100))
    return acc_list


# Plot as graph
acc_nomal = train_eval(x_train, x_test, is_peeky=False)
acc_rev = train_eval(x_train_rev, x_test_rev, is_peeky=False)
acc_pk = train_eval(x_train_rev, x_test_rev, is_peeky=True)
x_normal = np.arange(len(acc_nomal))
x_reverse = np.arange(len(acc_rev))
x_pk = np.arange(len(acc_pk))
plt.plot(x_normal, acc_nomal, marker='o', label="normal")
plt.plot(x_reverse, acc_rev, marker='v', label="reverse")
plt.plot(x_pk, acc_pk, marker='^', label="peeky+rev")

plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend()
plt.savefig('train_seq2seq.png')

