import pickle 
import os
import subprocess

def removeDS(array):    
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    return array

data_location = 'test/'
STORE_LOCATION = '/home/akshatgu/Intents-Analysis/Phone_Data' 

os.chdir(data_location)
files = os.listdir()
files = removeDS(files)

phones = []
for audio in files:
    result = subprocess.check_output('python3 -m allosaurus.run --lang hin -i ' + audio, shell=True)
    result_list = result.split()
    phones.append(result_list)

os.chdir(STORE_LOCATION)
a_file = open("data_synthesized.pkl", "wb")
pickle.dump(phones, a_file)
a_file.close()
