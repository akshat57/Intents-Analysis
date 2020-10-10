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


lang1 = 'hindi'
lang2 = 'marathi'
lang3 = 'bengali'
train_lang = lang1 + '_' + lang2 + '_' + lang3
test_lang = 'gujarati'
ratio = 0.5
filename = test_lang + '_' + f'{int(ratio*100):02}'
print(train_lang, test_lang, ratio, filename)


print('='*20, filename + '\n')
info_location = '/home/akshatgu/TaskMaster/Combinations/3_lang_variations/' + train_lang + '/'
data_location = '/home/akshatgu/TaskMaster/Combinations/3_lang_variations/' + train_lang + '/' + filename + '/'
STORE_LOCATION = '/home/akshatgu/Intents-Analysis/TaskMasterData/Get_Phones_Combos/3_lang_variations' 
index_to_lang = load_data(info_location + filename + '.pkl')
index_to_intents = load_data('index_to_intents.pkl')

count_lang1 = list(index_to_lang.values()).count(lang1)
count_lang2 = list(index_to_lang.values()).count(lang2)
count_lang3 = list(index_to_lang.values()).count(lang3)
count_testlang = list(index_to_lang.values()).count(test_lang)
total_count = count_lang1 + count_lang2 + count_lang3 + count_testlang
actual_ratio = count_testlang / total_count

print(actual_ratio, count_lang1/total_count, count_lang2/total_count, count_lang3/total_count)

os.chdir(data_location)
files = os.listdir()
files = removeDS(files)
files.sort()

all_phones = {}
for i, audio in enumerate(files):
    #index = audio.split('_')[1].split('.')[0]
    index = audio.split('.')[0]
    lang_code = index_to_lang[audio][:3]
    print(i, index, lang_code)

    result = subprocess.check_output('python3 -m allosaurus.run --lang ' + lang_code + ' -i ' + data_location + audio, shell=True)
    result_list = result.split()
    phones_list = [phone.decode('utf-8') for phone in result_list]
    
    all_phones[index] = phones_list


#Save data
os.chdir(STORE_LOCATION)
a_file = open("phones_taskmaster_" + train_lang + '_' + filename + ".pkl", "wb")
pickle.dump(all_phones, a_file)
a_file.close()


#Save percentages
f = open(train_lang + '_' + filename + ".txt", "a")
f.write('test_lang:' + str(actual_ratio) + ' | lang1:' + str(count_lang1/total_count) + ' | lang2:' + str(count_lang2/total_count) + ' | lang3:' + str(count_lang3/total_count))
f.close()
