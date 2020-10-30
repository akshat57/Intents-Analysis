from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def get_domains():
    all_intents = ['increase', 'decrease', 'activate', 'deactivate', 'bring', 'change language']
    return all_intents

def get_intents():
    all_intents = [
        'activate|lamp',
        'activate|lights|bedroom',
        'activate|lights|kitchen',
        'activate|lights|none',
        'activate|lights|washroom',
        'activate|music',
        'bring|juice',
        'bring|newspaper',
        'bring|shoes',
        'bring|socks',
        'change language|Chinese',
        'change language|English',
        'change language|German',
        'change language|Korean',
        'change language|none',
        'deactivate|lamp',
        'deactivate|lights|bedroom',
        'deactivate|lights|kitchen',
        'deactivate|lights|none',
        'deactivate|lights|washroom',
        'deactivate|music',
        'decrease|heat|bedroom',
        'decrease|heat|kitchen',
        'decrease|heat|none',
        'decrease|heat|washroom',
        'decrease|volume',
        'increase|heat|bedroom',
        'increase|heat|kitchen',
        'increase|heat|none',
        'increase|heat|washroom',
        'increase|volume'
        ]

    return all_intents


if __name__ == '__main__':
    type = 'domain'
    split = 'train'

    if type == 'domain':
        all_intents = get_domains()
    else:
        all_intents = get_intents()

    build_file = '../FSC/fsc_' + type + '_' + split + '.pkl'
    test_file = '../FSC/fsc_' + type + '_test.pkl'

    print(build_file)
    print(test_file)
    print('='*20)
    for ngram in range(3,4):
        print('='*30)
        print('')
        print('------------NGRAMS :', ngram, '\n')
        #for threshold in range(6):
        for threshold in [0, 2, 5, 10, 20]:
            
            blockPrint()
            frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
            correct, total, accuracy_per_intent = run_naive_bayes(frequency, word_index, ngram, test_file, all_intents)

            enablePrint()
            print('--THRESHOLD', threshold, ': ACCURACY = ', correct/total)
            print(accuracy_per_intent, '\n')

