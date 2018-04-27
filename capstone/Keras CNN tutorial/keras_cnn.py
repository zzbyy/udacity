from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Embedding
from keras.models import Model

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# load embedding vectors
print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))


# prepare text samples and labels
print('Preparing text dataset.')

train = fetch_20newsgroups(subset='train',
                           remove=('header', 'quotes', 'footer'))
test = fetch_20newsgroups(subset='test',
                          remove=('header', 'quotes', 'footer'))

X_train, X_test = train.data, test.data
y_train, y_test = train.target, test.target


def preprocess(docs):
    '''Tokenization, stemming and stopwords.'''
    tokenizer = RegexpTokenizer('[a-zA-Z]{2,}')
    # tokenizer = RegexpTokenizer('[\w\']')
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')

    new_docs = []

    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
        new_docs.append(' '.join(tokens))

    return new_docs


X_train = preprocess(X_train)
X_test = preprocess(X_test)

y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))


def get_vector(texts):
    # vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index

    embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


X_train_vector = get_vector(X_train)
X_test_vector = get_vector(X_test)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            weights=[X_train_vector],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')


# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(y_train), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(X_train_vector, y_train,
          batch_size=128,
          epochs=10,
          validation_split=0.2)


# evaluate mode
def evaluate(model):
    y_pred = model.predict(X_test_vector)
    # np.argmax(y_pred, axis=2) <==> np.array([np.argmax(arr, axis=1) for arr in y_pred])
    y_pred = np.argmax(y_pred, axis=2)
    y_true = np.argmax(y_test, axis=2)
    acc = np.mean(y_pred == y_true, axis=0)
    return acc


evaluate(model)
