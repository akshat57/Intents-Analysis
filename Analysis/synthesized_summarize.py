import pickle
import numpy as np

def list_to_string(input_string):
    output = ''
    for l in input_string:
        output += l
    return output

def summarize(synth_file = 'synthesized'):
    data_file = '../Phone_Data/data_' + synth_file + '.pkl'
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

    out_file = 'Labels/' + synth_file + '_hindi_labels.txt'
    with open(out_file, 'w') as filehandle:
        for sample in final_data:
            filehandle.write('%s\n' % sample)


if __name__ == '__main__':
    synth_file = 'synthesized_female_2'

    summarize(synth_file)


