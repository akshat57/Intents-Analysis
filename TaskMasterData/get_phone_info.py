import pickle 
import os
import subprocess

def removeDS(array):    
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array

info_location = '/home/akshatgu/TaskMaster/'

a_file = open( info_location + 'intent_stats.pkl', "rb")
intent_stats = pickle.load(a_file)
a_file.close()

a_file = open( info_location + 'labels.pkl', "rb")
labels = pickle.load(a_file)
a_file.close()

a_file = open( info_location + 'index_to_intents.pkl', "rb")
index_to_intents = pickle.load(a_file)
a_file.close()

a_file = open( 'data_taskmaster_103.pkl', "rb")
all_data = pickle.load(a_file)
a_file.close()

num_char = 0
for key in all_data:
        for utterance in all_data[key]:
                    num_char += len(utterance)


print('\n' + '='*20)
print('STATS:' )
print(intent_stats)
print('\nTOTAL NUMBER OF UTTERANCES:', sum(list(intent_stats.values())))
print('\nTotal Phones:', num_char)
print('\nAvg phone per utterances:', num_char / sum(list(intent_stats.values())))
print('='*20, '\n')


