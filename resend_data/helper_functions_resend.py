import subprocess
import pickle
import os

def removeDS(array):
    if '.DS_Store' in array:
        array.remove('.DS_Store')

    return array


def create_new_person(current_location, new_data_location, person):
    os.chdir(new_data_location)
    os.mkdir(person)
    os.chdir(current_location)


def create_new_conversation(current_location, new_data_location, person, valid_conversation_counter):
    '''This function creates a new conversation folder and copies all the audio files to new_data_location'''

    f = open('conversation_list.txt', 'r')
    audio_files = f.read().splitlines()
    f.close()

    #Create new conversation folder in new_data_location
    os.chdir(new_data_location + person)
    new_conversation_folder = 'conversation' + str(valid_conversation_counter)
    os.mkdir(new_conversation_folder)
    os.chdir(new_conversation_folder)

    #Transfering audio files
    for audio in audio_files:
        file_location = current_location + '/' + audio
        os.system('cp ' + file_location + ' ' + audio )

    #Transfering conversation_list.txt
    conversation_file_location = current_location + '/conversation_list.txt' 
    os.system('cp ' + conversation_file_location + ' conversation_list.txt') 
    
    os.chdir(current_location)
   
