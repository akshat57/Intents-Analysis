import pickle
import numpy as np

def list_to_string(input_string):
    output = ''
    for l in input_string:
        output += l.decode('utf-8')

    return output

def give_comparison(synth_file = 'synthesized', comparision_file = ''):
    filename = 'Labels/' + synth_file + 'hindi_labels.txt'
    f = open(filename, "r")
    synth_data = f.read().splitlines()
    f.close()

    filename = 'Labels/' + comparision_file + 'hindi_labels.txt'
    f = open(filename, "r")
    data = f.read().splitlines()
    f.close()

    accuracy = []
    for i, line in enumerate(data):
        synth_row = synth_data[i].split(',')
        comparison_row = line.split(',')
        
        phones_natural = [word for word in comparison_row[-1]]
        phones_synthesized = [word for word in synth_row[-1]]
        intersection = set(phones_synthesized).intersection(set(phones_natural))
        
        accuracy.append(len(intersection)/max(len(set(phones_synthesized)), len(set(phones_natural))) )
        
        print('Natural Phones:', len(set(phones_natural)))
        print('Synthesized Phones:', len(set(phones_synthesized)))
        print('Intersection:', len(intersection))
        print('')

    print('Mean similarity:', np.mean(np.array(accuracy)))
    print('Std similarity:', np.std(np.array(accuracy)))

if __name__ == '__main__':
    give_comparison('synthesized_female_1_', 'synthesized_female_2_')


