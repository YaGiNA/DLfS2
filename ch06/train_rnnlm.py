import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm


# Setting hyper paramater
batch_size = 20
wordvec_size = 100
hidden_size = 100   # Amount of RNN's hidden vector
time_size = 35      # Amount of RNN's size
lr = 20.0
max_epoch = 4
max_grad = 0.25

# Reading train data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# Generating model
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 1. Train w/ grad. clipping
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0, 500))

# 2. Evaluate by test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test preplexity: ', ppl_test)

# 3. Saving paramaters
model.save_params()

