import pickle
import numpy as np

def list_to_string(input_string):
    output = ''
    for l in input_string:
        output += l.decode('utf-8')

    return output

data_file = '../Phone_Data/data_synthesized.pkl'
a_file = open( data_file, "rb")
output = pickle.load(a_file)
a_file.close()
for i in range(len(output)):
    output[i] = [word.decode('utf-8') for word in output[i]]

filename = "Labels/hindi_labels.txt"
f = open(filename, "r")
data = f.read().splitlines()
f.close()

print(len(data))

accuracy = []
for i, line in enumerate(data):
    columns = line.split(',')
    phones_natural = [word for word in columns[-1]]
    phones_synthesized = [word for word in output[i]]
    intersection = set(phones_synthesized).intersection(set(phones_natural))
    
    accuracy.append(len(intersection)/max(len(set(phones_synthesized)), len(set(phones_natural))) )
    
    print('Natural Phones:', len(set(phones_natural)))
    print('Synthesized Phones:', len(set(phones_synthesized)))
    print('Intersection:', len(intersection))
    print('')

print('Mean similarity:', np.mean(np.array(accuracy)))
print('Std similarity:', np.std(np.array(accuracy)))

