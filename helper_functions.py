import subprocess 
import pickle
import os

def removeDS(array):
    if '.DS_Store' in array:
        array.remove('.DS_Store')
    
    return array


def get_labels( location = ''):
    f = open('conversation_list.txt', 'r')
    audio_files = f.read().splitlines()
    f.close()

    unique_intents = {}
    intent_list = []
    for audio in audio_files:
        f = open(audio + '_info')
        info = f.read()
        text = info.splitlines()[1].split(':')[1].lstrip()
        intent = info.splitlines()[3].split(':')[1].lstrip()
        f.close()

        if len(list(unique_intents.keys())) == 0:
            unique_intents[audio] = {'location': location, 'intent' : intent, 'text': text}
        elif intent != intent_list[-1]:
            unique_intents[audio] = {'location': location, 'intent' : intent, 'text': text}

        intent_list.append(intent)

    print(unique_intents.keys())
    for audio in unique_intents:
        result = subprocess.check_output('python3 -m allosaurus.run --lang eng -i ' + audio, shell=True)
        #result = subprocess.check_output('python3 -m allosaurus.run --lang hin -i ' + audio, shell=True)
        result_list = result.split()
        unique_intents[audio]['phones'] = result_list

    return unique_intents

if __name__ == '__main__':
    os.chdir('Data/akshatgu/conversation1')
    os.system('ls')
    print(get_labels())
