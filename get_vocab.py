import pickle

def load_data():
    a_file = open("intent_labels.pkl", "rb")
    output = pickle.load(a_file)
    a_file.close()

    return output

def build_ngrams(dialogue, N):

    vocab = []
    if len(dialogue) > 0:
        for i in range(N-1):
            dialogue.insert(0,'0'.encode('utf-8'))
            dialogue.append('0'.encode('utf-8'))
        
        for i in range(len(dialogue) -2):
            vocab.append(dialogue[i].decode('utf-8') + dialogue[i+1].decode('utf-8') + dialogue[i+2].decode('utf-8'))

        return vocab

    else:
        return vocab


def get_vocab(N = 3):
    data = load_data()

    vocab = []
    for key in data:
        for dialogue in data[key]:
            vocab += build_ngrams(dialogue, 3)

    return set(vocab)


if __name__ == '__main__':
    vocab = get_vocab(3)
    print(len(vocab), vocab)