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
for key in labelled_data:
    stats[key] = len(labelled_data[key])

print(stats)



