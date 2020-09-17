import pickle 
import os
import subprocess

def removeDS(array):    
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array

info_location = 'index_to_intents.pkl'
data_location = 'Phone_From_Text/phone_from_text_'
store_location = 'data_taskmaster_phoneFromText_' 

a_file = open( info_location, "rb")
index_to_intents = pickle.load(a_file)
a_file.close()

for lang in ['hindi', 'english', 'gujarati']:
    print(lang)
    all_phones = {}
    file = open(data_location + lang + '.txt', 'r')
    data = file.readlines()

    for line in data:
        line_split = line.split()
        index = line_split[0][-4:]
        phones_list = line_split[1:]

        if index_to_intents[index] in all_phones:
            all_phones[index_to_intents[index]].append(phones_list)
        else:
            all_phones[index_to_intents[index]] = [phones_list]

    #store data as pickle file
    a_file = open(store_location + lang + '.pkl', "wb")
    pickle.dump(all_phones, a_file)
    a_file.close()