import os
import pickle
from datetime import datetime
from helper_functions_resend import *


working_location = '/home/akshatgu/WorkingWithData/'
original_data_location =  working_location + 'Data/'
intent_data = {}

#Create New working directory
os.chdir(working_location)
now = datetime.now()
info = now.strftime("%Y%m%d_%H%M%S")
new_directory_name = 'Data_' + info
os.mkdir(new_directory_name)
new_data_location = working_location + new_directory_name + '/'


#Go to original_data_location
os.chdir(original_data_location)
people = os.listdir('.')
people = removeDS(people)


for person in people:
    print(person)

    #Go into a person
    person_location = person
    os.chdir(person_location)
    conversations = os.listdir('.')
    converstaions= removeDS(conversations)
    valid_conversation_counter = 0

    for conv in conversations:
        #Go into a conversation'''
        conv_location = conv
        os.chdir(conv_location)
        inside = os.listdir('.')
        inside= removeDS(inside)

        if 'conversation_list.txt' in inside:
            valid_conversation_counter += 1
            current_location = os.getcwd()

            if valid_conversation_counter == 1:
                create_new_person(current_location, new_data_location, person)
                
            create_new_conversation(current_location, new_data_location, person, valid_conversation_counter)


        os.chdir('..')
    os.chdir('..')
