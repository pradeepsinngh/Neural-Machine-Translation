import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size):
        super(Encoder, self).__init__()
        self.rnn_size = rnn_size
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = LSTM(rnn_size, return_sequences=True, return_state=True)

    def call(self, sequence, states):
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.rnn_size]), tf.zeros([batch_size, self.rnn_size]))


class LuongAttention(tf.keras.Model):
    def __init__(self, rnn_size, attention_func):
        super(LuongAttention, self).__init__()
        self.attention_func = attention_func

        if attention_func not in ['dot', 'general', 'concat']:
            raise ValueError('Unknown attention score function! Must be either dot, general or concat.')

        if attention_func == 'general':
            # General score function
            self.wa = Dense(rnn_size)
        elif attention_func == 'concat':
            # Concat score function
            self.wa = Dense(rnn_size, activation='tanh')
            self.va = Dense(1)

    def call(self, decoder_output, encoder_output):
        if self.attention_func == 'dot':
            # Dot score function: decoder_output (dot) encoder_output
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
        elif self.attention_func == 'general':
            # General score function: decoder_output (dot) (Wa (dot) encoder_output)
            # decoder_output has shape: (batch_size, 1, rnn_size)
            # encoder_output has shape: (batch_size, max_len, rnn_size)
            # => score has shape: (batch_size, 1, max_len)
            score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        elif self.attention_func == 'concat':
            # Concat score function: va (dot) tanh(Wa (dot) concat(decoder_output + encoder_output))
            # Decoder output must be broadcasted to encoder output's shape first
            decoder_output = tf.tile(decoder_output, [1, encoder_output.shape[1], 1])

            # Concat => Wa => va
            # (batch_size, max_len, 2 * rnn_size) => (batch_size, max_len, rnn_size) => (batch_size, max_len, 1)
            score = self.va(self.wa(tf.concat((decoder_output, encoder_output), axis=-1)))

            # Transpose score vector to have the same shape as other two above
            # (batch_size, max_len, 1) => (batch_size, 1, max_len)
            score = tf.transpose(score, [0, 2, 1])

        # alignment a_t = softmax(score)
        alignment = tf.nn.softmax(score, axis=2)

        # context vector c_t is the weighted average sum of encoder output
        context = tf.matmul(alignment, encoder_output)

        return context, alignment


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, rnn_size, attention_func):
        super(Decoder, self).__init__()
        self.attention = LuongAttention(rnn_size, attention_func)
        self.rnn_size = rnn_size
        self.embedding = Embedding(vocab_size, embedding_size)
        self.lstm = tf.keras.layers.LSTM(rnn_size, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(rnn_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        # Remember that the input to the decoder
        # is now a batch of one-word sequences,
        # which means that its shape is (batch_size, 1)
        embed = self.embedding(sequence)

        # Therefore, the lstm_out has shape (batch_size, 1, rnn_size)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)

        # Use self.attention to compute the context and alignment vectors
        # context vector's shape: (batch_size, 1, rnn_size)
        # alignment vector's shape: (batch_size, 1, source_length)
        context, alignment = self.attention(lstm_out, encoder_output)

        # Combine the context vector and the LSTM output
        # Before combined, both have shape of (batch_size, 1, rnn_size),
        # so let's squeeze the axis 1 first
        # After combined, it will have shape of (batch_size, 2 * rnn_size)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)

        # lstm_out now has shape (batch_size, rnn_size)
        lstm_out = self.wc(lstm_out)

        # Finally, it is converted back to vocabulary space: (batch_size, vocab_size)
        logits = self.ws(lstm_out)

        return logits, state_h, state_c, alignment
