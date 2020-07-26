from get_vocab import *
from get_frequency import *
import math
import operator

N = 1
frequency, word_index = get_frequency(N)
vocab, _ = get_vocab(N)
vocab_length = len(vocab)


for key in frequency:
    total_words = sum(list(frequency[key].values()))
    
    for word in frequency[key]:
        frequency[key][word] = math.log( (frequency[key][word] + 1) / (total_words + vocab_length) )



data = load_data()

correct = 0
total = 0
for key in data:
    for utterance in data[key]:
        total += 1
        ngrams = build_ngrams(utterance, N)
        probability = {}
        for key_naive in data:
            probability[key_naive] = 0
            for word in ngrams:
                probability[key_naive] += frequency[key_naive][word_index[word]]

        probability = sorted(probability.items(), key=operator.itemgetter(1), reverse = True)

        if key == probability[0][0]:
            correct += 1
        print(key, '--', probability[0][0])

print('Accuracy : ', correct/total, 'Total : ', total, 'Correct : ', correct)