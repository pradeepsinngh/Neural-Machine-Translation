# Machine Translation

## Introduction
In this project, I have worked on an End-to-End Machine translation pipeline that accpets English text as input and return the French translation.

This pipeline can be divided into 3 parts -
1. Part 1: Preprocessing 
2. Part 2: Models
3. Part 3: Prediction

### Preprocessing
Neural Network don't accpet textual data as input, they need input in numerical format only. So, we will have to text into sequences of integers. We will use following preprocess methods in Keras:

- Tokenization: Tokenize the words into ids
- Padding: Add padding to make all the sequences the same length.

#### Tokenize

For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings. Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).

We can turn each character into a number or each word into a number. These are called character and word ids, respectively. Character ids are used for character level models that generate text predictions for each character. A word level model uses word ids that generate text predictions for each word. Word level models tend to learn better, since they are lower in complexity, so we'll use those.

Turn each sentence into a sequence of words ids using Keras's Tokenizer function. Use this function to tokenize english_sentences and french_sentences in the cell below.

Running the cell will run tokenize on sample data and show output for debugging.

#### Padding

When batching the sequence of word ids together, each sequence needs to be the same length. Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.

Make sure all the English sequences have the same length and all the French sequences have the same length by adding padding to the end of each sequence using Keras's pad_sequences function.


### Models

In this section, you will experiment with various neural network architectures. You will begin by training four relatively simple architectures.

    Model 1 is a simple RNN
    Model 2 is a RNN with Embedding
    Model 3 is a Bidirectional RNN
    Model 4 is an optional Encoder-Decoder RNN
    Model 5 is Bidirectional RNN with Embedding

After experimenting with the four simple architectures, you will construct a deeper architecture that is designed to outperform all four models.

### Prediction
