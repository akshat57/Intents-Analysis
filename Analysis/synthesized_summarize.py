import pickle
import numpy as np

def list_to_string(input_string):
    output = ''
    for l in input_string:
        output += l
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

final_data = []
for i, line in enumerate(data):
    columns = line.split(',')
    columns[-1] = list_to_string(output[i])
    new_string = ','.join(columns)
    final_data.append(new_string)

out_file = 'Labels/synthesized_hindi_labels.txt'
with open(out_file, 'w') as filehandle:
    for sample in final_data:
        filehandle.write('%s\n' % sample)


