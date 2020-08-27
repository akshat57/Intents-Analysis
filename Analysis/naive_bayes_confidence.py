from get_vocab import load_data
from get_frequency import *
from naive_bayes import build_naive_bayes
import sys, os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_naive_bayes_with_confidence(frequency, word_index , N = 1, filename = 'Labels/intent_labels.pkl', all_intents = None):
    '''
        Use this function to run the naive bayes built in 'build_naive_bayes' with a test set.
        frequency, word_index, N: from naive bayes training
        filename = test file
    '''
    if all_intents == None:
        all_intents = ['Check Last Transaction', 'CheckBalance', 'Send Money', 'Withdraw Money', 'Deposit']
        
    data = load_data(filename)
    correct = 0
    total = 0
    accuracy_per_intent = {}

    threshold_accurate = {}
    threshold_inaccurate = {}
    second_correct = {}
    
    correct_sum = []
    incorrect_sum = []
    in2_sum = []

    for key in data:
        accuracy_per_intent[key] = 0
        for utterance in data[key]:
            total += 1
            ngrams = build_ngrams(utterance, N)
            probability = {}
            for key_naive in all_intents:
                probability[key_naive] = 0
                for word in ngrams:
                    if word in word_index:
                        if word_index[word] in frequency[key_naive]:
                            probability[key_naive] += frequency[key_naive][word_index[word]]
                        else:
                            probability[key_naive] += frequency[key_naive]['UNK']
                    else:
                        probability[key_naive] += frequency[key_naive]['UNK']
            
            probability = sorted(probability.items(), key=operator.itemgetter(1), reverse = True)

            if key == probability[0][0]:
                accuracy_per_intent[key] += 1
                correct += 1
                correct_sum.append((probability[0][1] - probability[1][1])/-probability[5][1])
            elif key == probability[1][0]:
                in2_sum.append((probability[0][1] - probability[1][1])/-probability[5][1])
            else:
                incorrect_sum.append((probability[0][1] - probability[1][1])/-probability[5][1])

            if (probability[0][1] - probability[1][1])/-probability[5][1] < 0.02135733:
                if key == probability[0][0]:
                    if key not in threshold_accurate:
                        threshold_accurate[key] = 1
                    else:
                        threshold_accurate[key] += 1
                else:
                    if probability[0][0] not in threshold_inaccurate:
                        threshold_inaccurate[probability[0][0]] = 1
                    else:
                        threshold_inaccurate[probability[0][0]] += 1

                    if key == probability[1][0]:
                        if probability[0][0] not in second_correct:
                            second_correct[probability[0][0]] = 1
                        else:
                            second_correct[probability[0][0]] += 1


            #print((probability[0][1] - probability[1][1]))
            #print(key, '--', probability[0][0], probability[1][0], probability[0][1] - probability[1][1])
        #accuracy_per_intent[key] /= len(data[key])


    #print(correct_sum/correct, incorrect_sum/(total - correct - in2_counter), in2_sum/in2_counter )
    print('Accuracy : ', correct/total, 'Total : ', total, 'Correct : ', correct)
    print(threshold_accurate)
    print(threshold_inaccurate)
    print(second_correct)

    print(len(correct_sum), len(incorrect_sum), len(in2_sum))
    maximum = max(max(correct_sum), max(incorrect_sum), max(in2_sum))
    nbins = 20
    mybins = np.linspace(0, maximum, nbins, True)

    '''sns.distplot(correct_sum, kde = False, bins = mybins)
    sns.distplot(in2_sum, kde = False, bins = mybins)
    sns.distplot(incorrect_sum, kde = False, bins = mybins)
    plt.legend(['correct', 'incorrect with 2nd choice', 'incorrect'])
    #sns.distplot(in2_sum)
    plt.show()'''


    if all_intents == None:
        return correct, total
    else:
        return correct, total, accuracy_per_intent

if __name__ == '__main__':
    all_intents = ['movie-tickets', 'auto-repair', 'restaurant-table', 'pizza-ordering', 'uber-lyft', 'coffee-ordering']

    build_file = 'Labels/TaskMaster/taskmaster_training_hindi.pkl'
    test_file = 'Labels/TaskMaster/taskmaster_testing_hindi.pkl'

    ngram = 3
    threshold = 0
    frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
    correct, total, accuracy_per_intent = run_naive_bayes_with_confidence(frequency, word_index, ngram, test_file, all_intents)
