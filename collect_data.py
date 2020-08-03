from helper_functions import *
import os
import pickle


current_location = os.getcwd()
data_location = 'Data/'
intent_data = {}

#Adjust for data folder manually
os.chdir('..')

#Go to data_location
os.chdir(data_location)
people = os.listdir('.')
people = removeDS(people)

for person in people:
    print(person)

    #Go into a person
    person_location = person
    os.chdir(person_location)
    conversations = os.listdir('.')
    converstaions= removeDS(conversations)

    for conv in conversations:
        #Go into a conversation
        
        conv_location = conv
        os.chdir(conv_location)
        inside = os.listdir('.')
        inside= removeDS(inside)
        
        if 'conversation_list.txt' in inside:
            print('--', conv)
            location = data_location + person + '/' + conv + '/'
            temp = get_labels(location)
            intent_data.update(temp)
        
        os.chdir('..')
    os.chdir('..')

os.chdir(current_location)
a_file = open("data_english.pkl", "wb")
pickle.dump(intent_data, a_file)
a_file.close()
