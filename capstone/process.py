from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import os
import pickle


def fetch_raw_data(categories, remove):
    train = fetch_20newsgroups(data_home='./data/',
                               subset='train',
                               categories=categories,
                               remove=remove)
    test = fetch_20newsgroups(data_home='./data/',
                              subset='test',
                              categories=categories,
                              remove=remove)
    X_train, y_train = train.data, train.target
    X_test, y_test = test.data, test.target

    return X_train, y_train, X_test, y_test


def clean_head_foot(docs):
    '''
    Remove header lines except 'Subject' and 'Organization'
    '''
    clean_docs = []
    for doc in docs:
        head, split, tail = doc.partition('\n\n')

        # clean head
        clean_head = '\n'.join([line.strip().split(':')[-1]
                                for line in head.strip().split('\n')
                                if line.strip().split(':')[0]
                                in ('Subject', 'Organization')])

        # remove foot
        splited_tail = tail.strip().split('\n')
        for i in range(len(splited_tail) - 1, -1, -1):
            if splited_tail[i] == '' or \
                    splited_tail[i].strip('-') == '' or \
                    splited_tail[i].strip('=') == '' or \
                    splited_tail[i].strip('#') == '' or \
                    splited_tail[i].strip('*') == '' or \
                    splited_tail[i].strip('\n') == '':
                break
        clean_tail = '\n'.join(splited_tail[:i])

        clean_doc = clean_head + '\n' + clean_tail
        clean_docs.append(clean_doc)

    return clean_docs


def preprocess(docs):
    '''Tokenization, stemming and stopwords.'''
    tokenizer = RegexpTokenizer('[a-zA-Z]{2,}')
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')

    new_docs = []

    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
        new_docs.append(' '.join(tokens))

    return new_docs


def load_data():
    '''Load cleaned data from preprocessed local data file,
    or clean data right now.'''
    data_file = './data/data.pkl'
    if os.path.exists(data_file):
        print('Loading cleaned data from file...')
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        return X_train, y_train, X_test, y_test
    else:
        print('Cleaning raw data...')
        data = {}
        X_train, y_train, X_test, y_test = fetch_raw_data(None, ('quotes'))

        X_train = clean_head_foot(X_train)
        X_train = preprocess(X_train)
        X_test = clean_head_foot(X_test)
        X_test = preprocess(X_test)

        data = {'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test}

        print('Storing cleaned data to data file...')
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)

        return X_train, y_train, X_test, y_test
