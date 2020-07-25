import os
import random
from datetime import datetime
import sox
from helper_functions_resend import *


resend_location = '/home/akshatgu/WorkingWithData/Data_Resent/'

#Go to working directory
os.chdir(resend_location)
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
        print('-',conv)
        session_id = str(random.randint(1,1000))
        #Go into a conversation : All conversations contain audio
        conv_location = conv
        os.chdir(conv_location)

        f = open('conversation_list.txt', 'r')
        audio_files = f.read().splitlines()
        f.close()
        
        for audio in audio_files:
            print('---', audio, os.getcwd())
            code = '/home/akshatgu/Intents-Analysis/resend_data/conversation.py'
            sample_rate = str(int(sox.file_info.sample_rate(audio)))
            command = 'python3 ' + code + ' ' + audio + ' ' + sample_rate + ' ' + session_id
            os.system(command)

        os.chdir('..')
    os.chdir('..')
