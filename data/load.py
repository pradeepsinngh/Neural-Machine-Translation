from data.preprocess import *
import io


def create_dataset(path, num_examples):
    '''
    Preprocess and return word pairs in the format: [ENGLISH, SPANISH]
    '''
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)


def load_dataset(path, num_examples=None):
    '''
    Creating cleaned input, output pairs.
    '''
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
