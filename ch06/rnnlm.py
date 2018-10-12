import sys
sys.path.append("..")
import numpy as np
from common.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from common.base_model import BaseModel
from typing import List


class Rnnlm(BaseModel):
    def __init__(self, vocab_size: int=10000, wordvec_size: int=100, hidden_size: int=100) -> None:
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # Initialize of weights
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D).astype('f'))
        lstm_Wh = (rn(D, 4 * H) / np.sqrt(H).astype('f'))
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # Generating layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # Conclude all of weights and grads as a list
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs: List[float]) -> List[float]:
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs: List[float], ts: List[float]) -> float:
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout: float=1) -> float:
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self) -> None:
        self.lstm_layer.reset_state()