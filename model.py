import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Bidirectional, Dense

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        gru = GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bigru = Bidirectional(gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state = hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self, batch):
        return tf.zeros((batch, 2*self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        gru = GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bigru = Bidirectional(gru, merge_mode='concat')
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1) # kaushik added this line
        output, forward_state, backward_state = self.bigru(x, initial_state = hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
