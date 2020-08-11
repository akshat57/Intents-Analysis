'''Here we train naive bayes on one synthesized'''
from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

print('TRAINED ON ENTIRE ONE (SYNTHETIC) DATA SET, TESTED ON ENTIRE ANOTHER (NATURAL) HINDI DATA\n')

for ngram in range(1,4):
    print('Ngram :', ngram)
    for threshold in range(5):

        build_file = 'Labels/intent_synthesized_hindi_labels.pkl'
        test_file = 'Labels/intent_hindi_labels.pkl'
        
        blockPrint()
        frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
        correct, total = run_naive_bayes(frequency, word_index, ngram, test_file)

        enablePrint()
        print('--Threshold', threshold, ': Accuracy = ', correct/total)

    print('')
