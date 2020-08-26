import pickle
import argparse

def preprocess(lang = 'hindi', flag = 'Train'):
    filename = "Labels/" + lang + "_labels.txt"
    f = open(filename, "r")
    data = f.read().splitlines()
    f.close()

    labelled_data = {}
    for line in data:
        elements = line.split(',')
        key = elements[2].strip()
        value = elements[4].strip()
        value = [x for x in value]
        
        if key in labelled_data:
            labelled_data[key].append(value)
        else:
            labelled_data[key] = [value]

    #Storing data with labels
    if flag == 'Train':
        a_file = open("Labels/intent_labels.pkl", "wb")
        pickle.dump(labelled_data, a_file)
        a_file.close()

    else:
        a_file = open("Labels/intent_" + lang + "_labels.pkl", "wb")
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

    mean_num_phones = sum(phone_lengths) / len(phone_lengths)
    print('\nTotal phones in the dataset : ', len(list_of_phones))
    print('Average number of phones per dialogue :', mean_num_phones)
    print('Number of unique phones : ', len(set(list_of_phones)))

    print('')
    return set(list_of_phones)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whatever')
    parser.add_argument('--lang', dest='lang', default='hindi', nargs='?')
    parser.add_argument('--flag', dest='flag', default='Train', nargs='?')
    args = parser.parse_args()

    preprocess(args.lang, args.flag)
    
