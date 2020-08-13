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

print(intent_stats)
print(labels)
print(index_to_intents)
