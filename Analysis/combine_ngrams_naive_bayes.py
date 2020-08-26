'''Here we combine best versions of unigrams, bigrams and trigrams'''
from get_vocab import load_data, build_ngrams
from naive_bayes import build_naive_bayes, run_naive_bayes, save_data
import operator

def cross_validation_combine(filename = "Labels/intent_labels.pkl", output_location = 'Labels/Combine/', total_test_sets = 12):
    data = load_data(filename)

    #allowed states are states with more than 1 training sample
    allowed_states = ['Check Last Transaction', 'CheckBalance', 'Send Money']
    yes = [True,True]
    no = [False,False]

    for i in range(total_test_sets):

        flag = no * i + yes + no * (total_test_sets-i)
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

        save_data(output_location + 'training_data_' + str(i) + '.pkl', training_data)
        save_data(output_location + 'testing_data_' + str(i) + '.pkl', testing_data)

def run_naive_bayes_single_utterance(frequency, word_index , N,utterance):
    '''Here we run naive bayes for a single utterence'''

    all_intents = ['Check Last Transaction', 'CheckBalance', 'Send Money', 'Withdraw Money', 'Deposit']
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
    
    #probability = sorted(probability.items(), key=operator.itemgetter(1), reverse = True)
    return probability

if __name__ == '__main__':
    create_dataset = 0
    if create_dataset:
        cross_validation_combine('Labels/intent_synthesized_female_2_hindi_labels.pkl', 'Labels/Combine_Ngram_Synth/')



    uni_correct = 0
    correct = 0
    total = 0
    for i in range(1):
        #training_data = 'Labels/Combine/training_data_' + str(i) + '.pkl'
        #testing_data = 'Labels/Combine/testing_data_' + str(i) + '.pkl'
        #training_data = 'Labels/Combine_Ngram_Synth/training_data_' + str(i) + '.pkl'
        #testing_data = 'Labels/Combine_Ngram_Synth/testing_data_' + str(i) + '.pkl'
        #training_data = 'Labels/Combine_Nat_Synth/test_synth/training_data_' + str(i) + '.pkl'
        #testing_data = 'Labels/Combine_Nat_Synth/test_synth/testing_data_' + str(i) + '.pkl'
        
        #training_data = 'Labels/intent_synthesized_hindi_labels.pkl'
        #training_data = 'Labels/intent_synthesized_all_female_hindi_labels.pkl'
        #training_data = 'Labels/intent_synthesized_all_hindi_labels.pkl'
        #testing_data = 'Labels/intent_hindi_labels.pkl'
        
        frequency_uni, word_index_uni = build_naive_bayes(1, training_data, 3)
        frequency_bi, word_index_bi = build_naive_bayes(2, training_data, 1)
        frequency_tri, word_index_tri = build_naive_bayes(3, training_data, 0)

        data = load_data(testing_data)
        for key in data:
            for utterance in data[key]:
                total += 1
                prob_uni = run_naive_bayes_single_utterance(frequency_uni, word_index_uni , 1, utterance)
                prob_bi = run_naive_bayes_single_utterance(frequency_bi, word_index_bi , 2, utterance)
                prob_tri = run_naive_bayes_single_utterance(frequency_tri, word_index_tri , 3, utterance)

                solution = {}
                for key_sol in prob_uni:
                    solution[key_sol] = prob_uni[key_sol] + prob_bi[key_sol] + prob_tri[key_sol]

                solution_sorted= sorted(solution.items(), key=operator.itemgetter(1), reverse = True)
                prob_uni_sorted = sorted(prob_uni.items(), key=operator.itemgetter(1), reverse = True)

                if key == solution_sorted[0][0]:
                    correct += 1

                if key == prob_uni_sorted[0][0]:
                    uni_correct +=1

                print(key)
                print('--Unigram:', prob_uni_sorted[0][0])
                print('--Combo:', solution_sorted[0][0])
                print('')

    print('Uni- Accuracy:', uni_correct/total)
    print('Combo Accuracy:', correct/total)



    #cross_validation_combine()