import sys
sys.path.insert(1, '/Users/manjugupta/Desktop/CMU_Courses/Intents/getting_intents/Analysis')

from get_vocab import load_data
from naive_bayes import save_data

data_location = '3_lang_variations/'
languages = 'hindi_marathi_bengali_gujarati_' + '50'
phone_file = data_location + 'phones_taskmaster_' + languages + '.pkl'

train_list = load_data('train_indices.pkl')
phone_data = load_data(phone_file)

train_data = {}
for intent in train_list:
    for index in train_list[intent]:
        if intent not in train_data:
            train_data[intent] = [phone_data[index]]
        else:
            train_data[intent].append(phone_data[index])

save_data( data_location + 'taskmaster_training_' + languages + '.pkl', train_data)
