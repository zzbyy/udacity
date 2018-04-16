from sklearn.datasets import fetch_20newsgroups


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
    # remove header lines except 'Subject' and 'Organization'

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
