from get_vocab import *
import matplotlib.pyplot as plt

def word_to_index(vocab):
    word_index = {}

    for i, word in enumerate(vocab):
        word_index[word] = i

    return word_index


if __name__ == '__main__':
    data = load_data()
    vocab = get_vocab()
    word_index = word_to_index(vocab)

    frequency = {}

    for key in data:
        frequency[key] = {}
        for word in vocab:
            frequency[key][word_index[word]] = 0

        for dialogue in data[key]:
            words = build_ngrams(dialogue, 3)
            for word in words:
                frequency[key][word_index[word]] += 1

    legend = []
    for key in data:  
        plt.plot(list(frequency[key].keys()), list(frequency[key].values()))
        legend.append(key)
    plt.legend(legend)
    plt.show()



