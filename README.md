# Machine Translation

## Introduction
In this project, I have worked on an End-to-End Machine translation pipeline that accpets English text as input and return the French translation.

This pipeline can be divided into 3 parts -
1. Part 1: Preprocessing 
2. Part 2: Models
3. Part 3: Prediction

### Preprocessing

For a neural network to predict on text data, it first has to be turned into data it can understand. Text data like "dog" is a sequence of ASCII character encodings. Since a neural network is a series of multiplication and addition operations, the input data needs to be number(s).

So, we will have to convert text into sequences of integers. We will use following methods in Keras:

- Tokenization: Tokenize the words into ids
- Padding: Add padding to make all the sequences the same length.

#### Tokenize

We can turn each character into a number or each word into a number. These are called character and word ids, respectively. Character ids are used for character level models that generate text predictions for each character. A word level model uses word ids that generate text predictions for each word. Word level models tend to learn better, since they are lower in complexity, so we'll use those.

We can turn each sentence into a sequence of words ids using [Keras's Tokenizer function](https://keras.io/preprocessing/text/#tokenizer).

#### Padding

When batching the sequence of word ids together, each sequence needs to be the same length. Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.

All the English sequences and the French sequences have the same length by adding padding to the end of each sequence using [Keras's pad_sequences function](https://keras.io/preprocessing/sequence/#pad_sequences).

### Models

Experimenting with various neural network architectures. I'll be training these five simple architectures:

    Model 1 is a simple RNN
    Model 2 is a RNN with Embedding
    Model 3 is a Bidirectional RNN
    Model 4 is a Encoder-Decoder RNN
    Model 5 is a Bidirectional RNN with Embedding (Final Model)

After experimenting with the four simple architectures, you will construct a deeper architecture that is designed to outperform all four models.

### Prediction
