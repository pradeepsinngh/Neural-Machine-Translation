import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras import Model

class Encoder(Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Encoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(lstm_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size]))


class Decoder(Model):
    def __init__(self, vocab_size, embedding_size, lstm_size):
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)

    def call(self, sequence, state):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)
        return logits, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size]))
