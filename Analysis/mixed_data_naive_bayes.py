'''Here we have previously mixes synth and natural datasets in the code mix_nat_synth_data.py'''
''' IN GENERAL, We're testing performance of Naive bayes on pre-defined test and training set.'''

from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

flag = 'synth'
file_location = 'Labels/Combine_Nat_Synth/test_' + flag + '/'
if flag == 'mix':
    total_files = 23
else:
    total_files = 12


for ngram in range(1,4):
    print('Ngram :', ngram)
    for threshold in range(5):
        total = 0
        correct = 0
        for i in range(total_files):

            build_file = file_location + 'training_data_' + str(i) + '.pkl'
            test_file = file_location + 'testing_data_' + str(i) + '.pkl'
        
            blockPrint()
            frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
            c, t = run_naive_bayes(frequency, word_index, ngram, test_file)

            correct += c
            total += t

            enablePrint()
        print('--Threshold', threshold, ': Accuracy = ', correct/total)

    print('')
