import pickle

a_file = open("data.pkl", "rb")
output = pickle.load(a_file)
a_file.close()

labelled_data = {}
for key in output:
    if output[key]['intent'] in labelled_data:
        labelled_data[output[key]['intent']].append(output[key]['phones'])
    else:
        labelled_data[output[key]['intent']] = [output[key]['phones']]

#Storing data with labels
a_file = open("intent_labels.pkl", "wb")
pickle.dump(labelled_data, a_file)
a_file.close()

stats = {}
phone_lengths = []
list_of_phones = []
for key in labelled_data:
    stats[key] = len(labelled_data[key])

    for utterance in labelled_data[key]:
        phone_lengths.append(len(utterance)) 
        list_of_phones += utterance

print('\nNumber of utterances per intent')
for key in stats:
    print('--', key, ':', stats[key])
print('')

mean_num_phones = sum(phone_lengths) / len(phone_lengths)
print('Total phones in the dataset : ', len(list_of_phones))
print('Total number of utterances :', len(phone_lengths))
print('Average number of phones per dialogue :', mean_num_phones)
print('Number of unique phones : ', len(set(list_of_phones)))

print('')



