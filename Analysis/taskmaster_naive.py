from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

all_intents = ['movie-tickets', 'auto-repair', 'restaurant-table', 'pizza-ordering', 'uber-lyft', 'coffee-ordering']

print("------TRAINED: guj, TESTED: hindi------")
for ngram in range(1,4):
    print('='*30)
    print('')
    print('------------NGRAMS :', ngram, '\n')
    for threshold in range(5):

        build_file = 'Labels/TaskMaster/taskmaster_training_guj.pkl'
        test_file = 'Labels/TaskMaster/taskmaster_testing_hindi.pkl'
        
        blockPrint()
        frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
        correct, total, accuracy_per_intent = run_naive_bayes(frequency, word_index, ngram, test_file, all_intents)

        enablePrint()
        print('--THRESHOLD', threshold, ': ACCURACY = ', correct/total)
        print(accuracy_per_intent, '\n')

