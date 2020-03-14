import tensorflow as tf
from network import Network


class model:
    def __init__(self):
        '''
        model initialization
        1. hyperparmeter setting
        2. load dataloader
        3. load networks(encoder, decoder, etc)
        '''
        pass

# comment @tf.function to debug
    @tf.function
    def train(self):
        self.train_epoch()
        self.test_epoch()
        pass

    @tf.function
    def train_epoch(self):
        pass

    @tf.function
    def test_epoch(self):
        pass

    @tf.function
    def evaluate(self):
        pass
