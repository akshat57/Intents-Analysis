import pickle 
import os
import subprocess

def removeDS(array):    
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array

info_location = '/home/akshatgu/TaskMaster/'
data_location = '/home/akshatgu/TaskMaster/google_TTS_US_ENG/'
STORE_LOCATION = '/home/akshatgu/Intents-Analysis/TaskMasterData' 

a_file = open( info_location + 'index_to_intents.pkl', "rb")
index_to_intents = pickle.load(a_file)
a_file.close()

os.chdir(data_location)
files = os.listdir()
files = removeDS(files)
files.sort()

all_phones = {}
for i, audio in enumerate(files):
    #index = audio.split('_')[1].split('.')[0]
    index = audio.split('.')[0]
    print(index)
    result = subprocess.check_output('python3 -m allosaurus.run --lang hin -i ' + data_location + audio, shell=True)
    result_list = result.split()
    phones_list = [phone.decode('utf-8') for phone in result_list]
    
    if index_to_intents[index] in all_phones: 
        all_phones[index_to_intents[index]].append(phones_list)
    else:
        all_phones[index_to_intents[index]] = [phones_list]


#Save data
os.chdir(STORE_LOCATION)
a_file = open("data_taskmaster_google_tts_USeng.pkl", "wb")
pickle.dump(all_phones, a_file)
a_file.close()
