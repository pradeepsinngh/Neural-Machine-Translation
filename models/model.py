import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU, Embedding, Bidirectional, Dense

class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        '''
        Args:
            vocab_size: input language vocabulary
            embedding_dim: embeddig dimension
            enc_units: units
            batch_sz: batch size
        '''
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        gru = GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bigru = Bidirectional(gru, merge_mode='concat')

    def call(self, x, hidden):
        '''
        Args:
            x: input tensor
            hidden: initial state in LSTM

        Returns:
            output:
            state: hidden state and cell state used for next steps
        '''

        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state = hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self, batch):
        return tf.zeros((batch, 2*self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, method):
        '''
        Args:
            units: units
            method: different types of method for score function, [concat, general, dot]
        '''
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.method = method

    def call(self, query, values):
        """
        Args:
            dec_h_t: current target state (batch_size, 1, units)
            enc_h_s: all source states (batch_size, seq_len, units)

        Returns:
            context_vector: (batch_size, units)
            attention_weights
        """
        hidden_with_time_axis = tf.expand_dims(query, 1) # query

        if self.method == 'concat':
            score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
            #score = self.V(tf.nn.tanh(self.W1(dec_h_t + enc_h_s)))
        elif self.method == 'general':
            score = tf.matmul(self.W1(values), query, transpose_b=True)
        else:
            # if score method is -- 'dot product'
            score = tf.matmul(hidden_with_time_axis, values)

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, method):
        '''
        Args:
            vocab_size: target language vocabulary
            embedding_dim: embeddig dimension
            dec_units: units
            batch_sz: batch size
        '''
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)

        gru = GRU(self.dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.bigru = Bidirectional(gru, merge_mode='concat')
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units, method)

    def call(self, x, hidden, enc_output):
        '''
        Args:
            x:
            hidden:
            enc_output:

        Returns:
            x:
            state:
            attention_weights:
        '''
        x = self.embedding(x)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state = hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights
