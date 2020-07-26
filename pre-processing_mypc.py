import pickle

f = open("intent_recognition_labels.txt", "r")
data = f.read().splitlines()
f.close()

labelled_data = {}
for line in data:
    elements = line.split(':')
    key = elements[2].strip()
    value = elements[4].strip()
    value = [x for x in value]
    
    if key in labelled_data:
        labelled_data[key].append(value)
    else:
        labelled_data[key] = [value]

#Storing data with labels
a_file = open("intent_labels.pkl", "wb")
pickle.dump(labelled_data, a_file)
a_file.close()

print('\nTotal number of utterances :', len(data))

phone_lengths = []
list_of_phones = []
for key in labelled_data:
    print('--', key, ':', len(labelled_data[key]))

    for utterance in labelled_data[key]:
        phone_lengths.append(len(utterance)) 
        list_of_phones += utterance

print('')

mean_num_phones = sum(phone_lengths) / len(phone_lengths)
print('Total phones in the dataset : ', len(list_of_phones))
print('Average number of phones per dialogue :', mean_num_phones)
print('Number of unique phones : ', len(set(list_of_phones)))

print('')