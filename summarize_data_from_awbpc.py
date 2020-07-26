import pickle
import os
import operator

a_file = open("data.pkl", "rb")
output = pickle.load(a_file)
a_file.close()

transcript_to_phones = {}
print('')
for key in output:
    phones = ''
    for phone in output[key]['phones']:
        phones += phone.decode('utf-8')

    #os.system('cp ../' + output[key]['location'] + key + ' ../audio_files/')
    transcript_to_phones[key] = [output[key]['location'], output[key]['intent'], output[key]['text'], phones]

sorted_ttp = sorted(transcript_to_phones.items(), key=operator.itemgetter(0))

for audio in sorted_ttp:
    print(audio[0], ':', audio[1][0], ':', audio[1][1], ':', audio[1][2], ':', audio[1][3],)
print('')
