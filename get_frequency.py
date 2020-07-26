from get_vocab import *
import matplotlib.pyplot as plt
import operator

def word_to_index(vocab):
    word_index = {}

    for i, (word, _) in enumerate(vocab):
        word_index[word] = i

    return word_index

def get_frequency(N):
    data = load_data()
    vocab, all_occurences = get_vocab(N)

    occurences = {}
    for word in vocab:
        occurences[word] = all_occurences.count(word)
    sorted_occurences = sorted(occurences.items(), key=operator.itemgetter(1), reverse = True)

    word_index = word_to_index(sorted_occurences)

    frequency = {}

    for key in data:
        frequency[key] = {}
        for word in vocab:
            frequency[key][word_index[word]] = 0

        for dialogue in data[key]:
            words = build_ngrams(dialogue, N)
            for word in words:
                frequency[key][word_index[word]] += 1

    return frequency, word_index


def plot_all(frequency):
    legend = []
    for key in data:  
        pass
        plt.bar(list(frequency[key].keys()), list(frequency[key].values()))
        legend.append(key)
    
    plt.legend(legend)
    plt.show()

def plot_few(frequency):
    intents = ['CheckBalance', 'Send Money', 'Check Last Transaction', 'Withdraw Money', 'Deposit' ]

    key = intents[0]
    plt.bar(list(frequency[key].keys()), list(frequency[key].values()))
    legend.append(key)
    key = intents[2]
    plt.bar(list(frequency[key].keys()), list(frequency[key].values()))
    legend.append(key)
    key = intents[4]
    plt.bar(list(frequency[key].keys()), list(frequency[key].values()))
    legend.append(key) 

    plt.legend(legend)
    plt.show()   



if __name__ == '__main__':

    frequency, sorted_occurences = get_frequency(1)
    print(frequency)

