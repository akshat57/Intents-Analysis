from get_vocab import *
from get_frequency import *
import math
import operator

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


def build_naive_bayes(N=1, filename='intent_labels.pkl'):
    frequency, word_index = get_frequency(N, filename)
    vocab, _ = get_vocab(N, filename)
    vocab_length = len(vocab)

    for key in frequency:
        total_words = sum(list(frequency[key].values()))
        
        for word in frequency[key]:
            frequency[key][word] = math.log( (frequency[key][word] + 1) / (total_words + vocab_length) )
    

    return frequency, word_index


def run_naive_bayes(frequency, word_index , N = 1, filename = 'intent_labels.pkl'):

    data = load_data(filename)
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


def cross_validation(N = 1):
    data = load_data()

    #allowed states are states with more than 1 training sample
    allowed_states = ['Check Last Transaction', 'CheckBalance', 'Send Money']
    yes = [True,True]
    no = [False,False]

    for i in range(12):
        flag = no * i + yes + no * (11-i)
        state_counter = 0
        training_data = {}
        testing_data = {}

        for key in data:
            if key not in allowed_states:
                training_data[key] = data[key]
            else:
                for utterance in data[key]:
                    if flag[state_counter]:
                        if key in testing_data:
                            testing_data[key].append(utterance)
                        else:
                            testing_data[key] = [utterance]

                    else:
                        if key in training_data:
                            training_data[key].append(utterance)
                        else:
                            training_data[key] = [utterance]


                    state_counter +=1

        save_data('training_data.pkl', training_data)
        save_data('testing_data.pkl', testing_data)

        frequency, word_index = build_naive_bayes(N, 'training_data.pkl')
        #print('Training', '-'*20)
        #run_naive_bayes(frequency, word_index, N, 'training_data.pkl')
        #print('')
        print('Testing', '-'*20)
        run_naive_bayes(frequency, word_index, N, 'testing_data.pkl')
        print('='*50)
        print('')


cross_validation(2)

###USE THIS TO TRAIN WITH THE ENTIRE DATASET AND FIND ACCURACY ON TRAIN SET
#frequency, word_index = build_naive_bayes()
#run_naive_bayes(frequency, word_index)
