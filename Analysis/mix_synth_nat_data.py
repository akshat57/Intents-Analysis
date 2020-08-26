'''Here we are mixing up hindi natural speech data and synthesized hindi flag data.
    There are three cases:
        1. Testing only with natural data
        2. Testing only with synthesized data
        3. Testing with combined
'''
import pickle
from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes, save_data
import sys, os
from combine_ngrams_naive_bayes import cross_validation_combine

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__



def combine_nat_synth_testmix(set1, set2):
    set1_data = load_data(set1)
    set2_data = load_data(set2)

    combined_data = {}
    for key in set1_data:
        combined_data[key] = set1_data[key] + set2_data[key]
    
    a_file = open("Labels/intent_synthesized_all_hindi_labels.pkl", "wb")
    pickle.dump(combined_data, a_file)
    a_file.close()

    #cross_validation_combine('Labels/temp.pkl', 'Labels/Combine_Nat_Synth/test_mix/', 23)



def combine_nat_synth_test(synthetic_labels, natural_labels, test_flag):
    synthetic = load_data(synthetic_labels)
    natural = load_data(natural_labels)

    if test_flag == 'nat':
        data = natural
    else:
        data = synthetic

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


        combined_data = {}
        for key in natural:
            if test_flag == 'nat':
                combined_data[key] = training_data[key] +synthetic[key]
            else:
                combined_data[key] = training_data[key] + natural[key]

        save_data('Labels/Combine_Nat_Synth/test_' + test_flag + '/training_data_' + str(i) + '.pkl', combined_data)
        save_data('Labels/Combine_Nat_Synth/test_' + test_flag + '/testing_data_' + str(i) + '.pkl', testing_data)

    
    #a_file = open("Labels/temp.pkl", "wb")
    #pickle.dump(combined_data, a_file)
    #a_file.close()






###Mix data up
#synthetic_labels = 'Labels/intent_synthesized_hindi_labels.pkl'
#natural_labels = 'Labels/intent_hindi_labels.pkl'
#synthetic_labels = 'Labels/intent_synthesized_female_1_hindi_labels.pkl'
#natural_labels = 'Labels/intent_synthesized_all_female_hindi_labels.pkl'
#combine_nat_synth_testmix(synthetic_labels, natural_labels)
#combine_nat_synth_test(synthetic_labels, natural_labels, 'nat')
#combine_nat_synth_test(synthetic_labels, natural_labels, 'synth')