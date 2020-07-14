
from preprocessing import *
import tensorflow as tf
import io


def read_file(filename):
    with io.open(filename, encoding='UTF-8') as f:
        lines = f.read()
    return lines

def prepare_data(FILENAME, NUM_EXAMPLES):

    lines = read_file(FILENAME)

    raw_data = []
    for line in lines.split('\n'):
        raw_data.append(line.split('\t'))

    raw_data = raw_data[:-1]

    print(raw_data[:5])

    word_pairs = [[preprocess_sentence(w) for w in l[:2]]  for l in raw_data[:NUM_EXAMPLES]]

    raw_data_en, raw_data_fr = zip(*word_pairs)

    en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    en_tokenizer.fit_on_texts(raw_data_en)
    data_en = en_tokenizer.texts_to_sequences(raw_data_en)
    data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en, padding='post')
    print('English sequences')
    print(data_en[:2])

    fr_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    fr_tokenizer.fit_on_texts(raw_data_fr)
    data_fr = fr_tokenizer.texts_to_sequences(raw_data_fr)
    data_fr = tf.keras.preprocessing.sequence.pad_sequences(data_fr, padding='post')
    print('French input sequences')
    print(data_fr[:2])

    return data_en, data_fr, raw_data_en, raw_data_fr, en_tokenizer, fr_tokenizer
