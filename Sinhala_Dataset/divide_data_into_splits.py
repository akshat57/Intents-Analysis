import pickle
import csv
import random

def load_data(filename = "Labels/intent_labels.pkl"):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()


def read_file(filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        name_to_intent = {}
        intent_freq = {}

        for i, row in enumerate(readCSV):
            if i > 0:
                audio_file = row[0]
                intent =row[1]
                name_to_intent[audio_file] = intent

                #saving intent-wise data statistics
                if intent in intent_freq:
                    intent_freq[intent] +=1
                else:
                    intent_freq[intent] = 1


    return name_to_intent, intent_freq



def divide_into_cross_validation_split(audios, N = 5):
    random.shuffle(audios)
    random.shuffle(audios)

    split_size = len(audios) // 5
    test_splits = []

    for i in range(N):
        test_splits.append(audios[i*split_size : (i + 1)*split_size])

    return test_splits



data = load_data('/home/akshatgu/SpeechDatasets/Sinhala_Datset/phonemic_transcription/top5/Sinhala_phoneme_transcript.pkl')
name_to_intent, intent_freq = read_file('/home/akshatgu/SpeechDatasets/Sinhala_Datset/Sinhala_Data.csv')  

audios = list(data.keys())
test_splits = divide_into_cross_validation_split(audios)


for i, test_split in enumerate(test_splits):
    train_data = {}
    test_data = {}

    for filename in data:
        intent = name_to_intent[filename]
    
        #if file in test_split, store in test_data. Else in train data
        if filename in test_split:
            if intent in test_data:
                test_data[intent].append(data[filename])
            else:
                test_data[intent] = [data[filename]]


        else:
            if intent in train_data:
                train_data[intent].append(data[filename])
            else:
                train_data[intent] = [data[filename]]

    save_data('datasplit_top5_split1/sinhala_train_split_' + str(i+1) + '.pkl', train_data)
    save_data('datasplit_top5_split1/sinhala_test_split_' + str(i +1) + '.pkl', test_data)
