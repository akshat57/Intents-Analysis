from contact_tts import do_tts
import random
import pickle

def save_data(filename, data):
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


audio_save_location = '/home/akshatgu/TaskMaster/translated_hindi/'
language_code = 'hi-IN'
#voices = ['hi-IN-Standard-A', 'hi-IN-Standard-B', 'hi-IN-Standard-C', 'hi-IN-Standard-D']
voices = ['hi-IN-Standard-A', 'hi-IN-Standard-B']

#Read text
a_file = open( '../DATA_VERSION1/taskmaster_hindi_intent_text.txt', "r")
text = a_file.readlines()
a_file.close()

#Creating index to text dictionary
utterances= {}
for utt in text:
        index = utt.split('|')[0]
        u = utt.split('|')[1][:-1]
        utterances[index] = u

total_utt = len(list(utterances.keys()))
all_keys = list(utterances.keys())

#Creating index to voice name dictionary
index_to_voice = {}
while len(all_keys) != 0:
    key = random.choice(all_keys)
    all_keys.remove(key)
    index_to_voice[key] = random.choice(voices)

#Sending for TTS
for key in utterances:
    text = utterances[key]
    output_file = audio_save_location + key
    voice_name = index_to_voice[key]
    print(key, voice_name)
    do_tts(text, output_file, language_code, voice_name)

save_data('index_to_voice_hindi.pkl', index_to_voice)
