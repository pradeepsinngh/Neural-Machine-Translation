from model import *
from preprocessing import *
from utils import loss_func
from data import prepare_data

import tensorflow as tf
import numpy as np
import unicodedata
import re
import matplotlib.pyplot as plt
import os
import requests
import io


def predict(test_source_text=None):
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]

    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_initial_states = encoder.init_states(1)
    en_outputs = encoder(tf.constant(test_source_seq), en_initial_states)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    alignments = []

    while True:
        de_output, de_state_h, de_state_c, alignment = decoder(de_input, (de_state_h, de_state_c), en_outputs[0])
        de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
        out_words.append(fr_tokenizer.index_word[de_input.numpy()[0][0]])

        alignments.append(alignment.numpy())

        if out_words[-1] == '<end>' or len(out_words) >= 20:
            break

    print(' '.join(out_words))
    return np.array(alignments), test_source_text.split(' '), out_words


def train_step(source_seq, target_seq, en_initial_states):

    """ Execute one training step (forward pass + backward pass)
    Args:
        source_seq: source sequences
        target_seq: input target sequences (<start> + ... + <end>)

    Returns:
        The loss value of the current pass
    """

    loss = 0
    with tf.GradientTape() as tape:
        en_outputs = encoder(source_seq, en_initial_states)
        en_states = en_outputs[1:]
        de_state_h, de_state_c = en_states

        # We need to create a loop to iterate through the target sequences
        for i in range(target_seq.shape[1]):
            # Input to the decoder must have shape of (batch_size, length)
            # so we need to expand one dimension
            decoder_in = tf.expand_dims(target_seq[:, i], 1)
            logit, de_state_h, de_state_c, _ = decoder(decoder_in, (de_state_h, de_state_c), en_outputs[0])

            # The loss is now accumulated through the whole batch
            loss += loss_func(target_seq[:, i], logit)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss / target_seq.shape[1]



if __name__ == '__main__':

    # parameters
    MODE = 'TEST'
    FILENAME = 'data/fra-eng/fra.txt' #'/data/fra-eng.zip'
    NUM_EXAMPLES = 64 * 10
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    RNN_SIZE = 512
    NUM_EPOCHS = 11
    ATTENTION_FUNC = 'concat' # Can choose between 'dot', 'general' or 'concat'

    # prepare_data
    data_en, data_fr, raw_data_en, raw_data_fr, en_tokenizer, fr_tokenizer = prepare_data(FILENAME, NUM_EXAMPLES)

    # data set
    dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr))
    dataset = dataset.shuffle(len(raw_data_en)).batch(BATCH_SIZE)

    # get voculabry size
    en_vocab_size = len(en_tokenizer.word_index) + 1
    fr_vocab_size = len(fr_tokenizer.word_index) + 1

    encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, RNN_SIZE)
    decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, RNN_SIZE, ATTENTION_FUNC)

    # Intializing encoder and decoder
    initial_state = encoder.init_states(1)
    encoder_outputs = encoder(tf.constant([[1]]), initial_state)
    decoder_outputs = decoder(tf.constant([[1]]), encoder_outputs[1:], encoder_outputs[0])

    optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)

    if not os.path.exists('checkpoints/encoder'):
        os.makedirs('checkpoints/encoder')
    if not os.path.exists('checkpoints/decoder'):
        os.makedirs('checkpoints/decoder')

    if MODE == "TRAIN":

        for e in range(NUM_EPOCHS):
            en_initial_states = encoder.init_states(BATCH_SIZE)

            if e % 100 == 0:
                encoder.save_weights('checkpoints/encoder/encoder_{}.h5'.format(e + 1))
                decoder.save_weights('checkpoints/decoder/decoder_{}.h5'.format(e + 1))

            for batch, (source_seq, target_seq) in enumerate(dataset.take(-1)):
                loss = train_step(source_seq, target_seq, en_initial_states)

                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(e + 1, batch, loss.numpy()))

        encoder.save_weights(filepath='checkpoints/encoder/encoder_final.h5')
        decoder.save_weights(filepath='checkpoints/decoder/decoder_final.h5')

        print('Done Training...')

    else:
        # for testing/ predicting

        # load latest checkpoints
        encoder_checkpoint_path = 'checkpoints/encoder/encoder_final.h5'
        decoder_checkpoint_path = 'checkpoints/decoder/decoder_final.h5'
        print(encoder_checkpoint_path)
        print(decoder_checkpoint_path)
        print('-----------------------')
        # load weights
        encoder.load_weights(encoder_checkpoint_path)
        decoder.load_weights(decoder_checkpoint_path)

        test_sents = (
            'What a ridiculous concept!',
            'Your idea is not entirely crazy.',
            "A man's worth lies in what he is.",
            'What he did is very wrong.',
            "All three of you need to do that.",
            "Are you giving me another chance?",
            "Both Tom and Mary work as models.",
            "Can I have a few minutes, please?",
            "Could you close the door, please?",
            "Did you plant pumpkins this year?",
            "Do you ever study in the library?",
            "Don't be deceived by appearances.",
            "Excuse me. Can you speak English?",
            "Few people know the true meaning.",
            "Germany produced many scientists.",
            "Guess whose birthday it is today.",
            "He acted like he owned the place.",
            "Honesty will pay in the long run.",
            "How do we know this isn't a trap?",
            "I can't believe you're giving up.",
        )

        filenames = []

        for i, test_sent in enumerate(test_sents):
            test_sequence = normalize_string(test_sent)
            alignments, source, prediction = predict(test_sequence)
            attention = np.squeeze(alignments, (1, 2))
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(attention, cmap='jet')
            ax.set_xticklabels([''] + source, rotation=90)
            ax.set_yticklabels([''] + prediction)

            filenames.append('result/test_{}.png'.format(i))
            plt.savefig('result/test_{}.png'.format(i))
            plt.close()

        print('Done Predicting...')
