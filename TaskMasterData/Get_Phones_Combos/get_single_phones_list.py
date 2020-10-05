import pickle 
import os
import subprocess

def load_data(filename = "Labels/intent_labels.pkl"):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    
    return output


def removeDS(array):    
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array

filename = 'bengali'
print('='*10, filename)
info_location = '/home/akshatgu/TaskMaster/'
data_location = '/home/akshatgu/TaskMaster/translated_' + filename + '/'
STORE_LOCATION = '/home/akshatgu/Intents-Analysis/TaskMasterData/Get_Phones_Combos/1_language' 
index_to_intents = load_data('index_to_intents.pkl')

os.chdir(data_location)
files = os.listdir()
files = removeDS(files)
files.sort()


all_phones = {}
for i, audio in enumerate(files):
    
    index = audio.split('.')[0]
    print(i, index)

    result = subprocess.check_output('python3 -m allosaurus.run --lang ' + filename[:3] + ' -i ' + data_location + audio, shell=True)
    result_list = result.split()
    phones_list = [phone.decode('utf-8') for phone in result_list]
    
    all_phones[index] = phones_list

#Save data
os.chdir(STORE_LOCATION)
a_file = open("phones_taskmaster_" + filename + ".pkl", "wb")
pickle.dump(all_phones, a_file)
a_file.close()
