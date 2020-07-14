from model import *
from preprocessing import *
from utils import *
from data import prepare_data

import tensorflow as tf
import numpy as np
import unicodedata
import time, re, os, io
import matplotlib.pyplot as plt


def predict(test_source_text=None):
    """ Predict the output sentence for a given input sentence
    Args:
        test_source_text: input sentence (raw string)

    Returns:
        The encoder's attention vectors
        The decoder's bottom attention vectors
        The decoder's middle attention vectors
        The input string array (input sentence split by ' ')
        The output string array
    """
    if test_source_text is None:
        test_source_text = raw_data_en[np.random.choice(len(raw_data_en))]

    print(test_source_text)
    test_source_seq = en_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_output, en_alignments = encoder(tf.constant(test_source_seq),pes=pes, training=False)

    de_input = tf.constant([[fr_tokenizer.word_index['<start>']]], dtype=tf.int64)
    out_words = []

    while True:
        de_output, de_bot_alignments, de_mid_alignments = decoder(de_input, en_output,pes=pes, training=False)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(fr_tokenizer.index_word[new_word.numpy()[0][0]])

        # Transformer doesn't have sequential mechanism (i.e. states)
        # so we have to add the last predicted word to create a new input sequence
        de_input = tf.concat((de_input, new_word), axis=-1)

        # TODO: get a nicer constraint for the sequence length!
        if out_words[-1] == '<end>' or len(out_words) >= 14:
            break

    print(' '.join(out_words))
    return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words


def train_step(source_seq, target_seq):
    """ Execute one training step (forward pass + backward pass)
    Args:
        source_seq: source sequences
        target_seq: input target sequences (<start> + ... + <end>)

    Returns:
        The loss value of the current pass
    """

    loss = 0
    with tf.GradientTape() as tape:
        encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_output, _ = encoder(source_seq, pes=pes, encoder_mask=encoder_mask)
        decoder_output, _, _ = decoder(target_seq, encoder_output, pes=pes, encoder_mask=encoder_mask)
        loss = loss_func(target_seq, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


if __name__ == '__main__':

    MODE = 'TRAIN'
    FILENAME = 'data/fra-eng/fra.txt' #'/data/fra-eng.zip'
    NUM_EXAMPLES = 64 * 100
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    RNN_SIZE = 512
    NUM_EPOCHS = 25
    MODEL_SIZE = 128
    H = 8
    NUM_LAYERS = 4
    ATTENTION_FUNC = 'concat' # Can choose between 'dot', 'general' or 'concat'

    # prepare_data
    data_en, data_fr, raw_data_en, raw_data_fr, en_tokenizer, fr_tokenizer = prepare_data(FILENAME, NUM_EXAMPLES)
    max_length = max(len(data_en[0]), len(data_fr[0]))
    dataset = tf.data.Dataset.from_tensor_slices((data_en, data_fr))
    dataset = dataset.shuffle(len(raw_data_en)).batch(BATCH_SIZE)


    pes = []
    for i in range(max_length):
        pes.append(positional_encoding(i, MODEL_SIZE))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)

    print(pes.shape)
    print(data_en.shape)
    print(data_fr.shape)


    en_vocab_size = len(en_tokenizer.word_index) + 1
    fr_vocab_size = len(fr_tokenizer.word_index) + 1
    encoder = Encoder(en_vocab_size, MODEL_SIZE, NUM_LAYERS, H)
    decoder = Decoder(fr_vocab_size, MODEL_SIZE, NUM_LAYERS, H)

    sequence_in = tf.constant([[1, 2, 3, 0, 0]])
    encoder_output, _ = encoder(sequence_in, pes)
    encoder_output.shape

    sequence_in = tf.constant([[14, 24, 36, 0, 0]])
    decoder_output, _, _ = decoder(sequence_in, encoder_output, pes)
    decoder_output.shape

    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr = WarmupThenDecaySchedule(MODEL_SIZE)
    optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    if MODE == 'TRAIN':

        starttime = time.time()
        for e in range(NUM_EPOCHS):

            if e % 100 == 0:
                encoder.save_weights('checkpoints/encoder/encoder_{}.h5'.format(e + 1))
                decoder.save_weights('checkpoints/decoder/decoder_{}.h5'.format(e + 1))

            for batch, (source_seq, target_seq) in enumerate(dataset.take(-1)):
                loss = train_step(source_seq, target_seq)
                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(e + 1, batch, loss.numpy(), time.time() - starttime))


        encoder.save_weights(filepath='checkpoints/encoder/encoder_final.h5')
        decoder.save_weights(filepath='checkpoints/decoder/decoder_final.h5')

        print('Done Training ....')

    else:
        # for testing / predicting ..

        # load latest checkpoints
        encoder_checkpoint_path = 'checkpoints/encoder/encoder_final.h5'
        decoder_checkpoint_path = 'checkpoints/decoder/decoder_final.h5'
        print(encoder_checkpoint_path)
        print(decoder_checkpoint_path)
        print('-----------------------')
        # load weights
        encoder.load_weights(encoder_checkpoint_path)
        decoder.load_weights(decoder_checkpoint_path)


        test_sents = ('Did you plant pumpkins this year?',)

        '''
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
        '''

        for i, test_sent in enumerate(test_sents):
            test_sequence = normalize_string(test_sent)
            predict(test_sequence)
