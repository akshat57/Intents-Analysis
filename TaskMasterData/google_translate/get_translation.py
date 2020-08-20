from contact_translate import translate_text
import random
import pickle

def save_data(filename, data):
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


audio_save_location = '/home/akshatgu/TaskMaster/google_translate/'

#Read text
a_file = open( '../intent_text.txt', "r")
text = a_file.readlines()
a_file.close()

#Creating index to text dictionary
utterances= {}
for utt in text:
        index = utt.split('|')[0]
        u = utt.split('|')[1][:-1]
        utterances[index] = u

file1 = open('taskmaster_hindi_intent_text.txt','a')
hindi_translation = []
for key in utterances:
    if int(key) >= 0:
        print(key)
        translated_text = translate_text(utterances[key])
        row = key + '|' +  translated_text
    
        file1.write(row + '\n')
        hindi_translation.append(row)
    
file1.close()

with open('taskmaster_hindi_intent_text_full.txt', 'w') as filehandle:
    for row in hindi_translation:
        filehandle.write('%s\n' % row)

