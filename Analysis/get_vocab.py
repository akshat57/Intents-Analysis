import pickle

def load_data(filename = "Labels/intent_labels.pkl"):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def build_ngrams(dialogue, N):

    vocab = []
    if len(dialogue) > 0:
        for i in range(N-1):
            dialogue.insert(0,'0')
            dialogue.append('0')
        
        for i in range(len(dialogue) - N + 1):
            word = ''
            for j in range(N):
                word += dialogue[i+j]
            vocab.append(word)
 
        return vocab
    else:
        return vocab


def get_vocab(N = 3, filename= "Labels/intent_labels.pkl" ):
    data = load_data(filename)

    vocab = []
    for key in data:
        for dialogue in data[key]:
            vocab += build_ngrams(dialogue, N)

    return set(vocab), vocab


if __name__ == '__main__':
    print('\nNumber of unique N-gram phones : ')
    for i in range(1,5):
        vocab, _ = get_vocab(i)
        print('-- N =', i , ':', len(vocab))
        
    print('')
