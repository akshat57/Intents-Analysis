from get_vocab import *
from get_frequency import *
import math
import operator

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


def build_naive_bayes(N=1, filename='Labels/intent_labels.pkl'):
    '''
        N = N-gram
        filename = filename for training set
    '''
    frequency, word_index = get_frequency(N, filename)
    vocab, _ = get_vocab(N, filename)
    vocab_length = len(vocab)

    for key in frequency:
        total_words = sum(list(frequency[key].values()))
        
        for word in frequency[key]:
            frequency[key][word] = math.log( (frequency[key][word] + 1) / (total_words + vocab_length) )
        frequency[key]['UNK'] = 1 / (total_words + vocab_length) 

    return frequency, word_index


def run_naive_bayes(frequency, word_index , N = 1, filename = 'Labels/intent_labels.pkl'):
    '''
        Use this function to run the naive bayes built in 'build_naive_bayes' with a test set.
        frequency, word_index, N: from naive bayes training
        filename = test file
    '''
    all_intents = ['Check Last Transaction', 'CheckBalance', 'Send Money', 'Withdraw Money', 'Deposit']
    data = load_data(filename)
    correct = 0
    total = 0
    for key in data:
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
                correct += 1
            print(key, '--', probability[0][0])

    print('Accuracy : ', correct/total, 'Total : ', total, 'Correct : ', correct)
    
    return correct, total



def cross_validation(N = 1):
    data = load_data()

    #allowed states are states with more than 1 training sample
    allowed_states = ['Check Last Transaction', 'CheckBalance', 'Send Money']
    yes = [True,True]
    no = [False,False]

    correct = 0
    total = 0
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

        save_data('Labels/training_data.pkl', training_data)
        save_data('Labels/testing_data.pkl', testing_data)

        frequency, word_index = build_naive_bayes(N, 'Labels/training_data.pkl')
        #print('Training', '-'*20)
        #run_naive_bayes(frequency, word_index, N, 'Labels/training_data.pkl')
        #print('')
        print('Testing', '-'*20)
        metrics = run_naive_bayes(frequency, word_index, N, 'Labels/testing_data.pkl')
        print('='*50)
        print('')

        correct += metrics[0]
        total += metrics[1]

    print('Overal accuracy --', correct/total)


if __name__ == '__main__':
    cross_validation(3)


###USE THIS TO TRAIN WITH THE ENTIRE DATASET AND FIND ACCURACY ON TRAIN SET
    '''N = 3
    frequency, word_index = build_naive_bayes(N)
    run_naive_bayes(frequency, word_index, N)
'''