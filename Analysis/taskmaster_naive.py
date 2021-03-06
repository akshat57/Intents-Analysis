from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes
import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':

    all_intents = ['movie-tickets', 'auto-repair', 'restaurant-table', 'pizza-ordering', 'uber-lyft', 'coffee-ordering']
    build_file = '../TaskMasterData/Get_Phones_Combos/3_languages/taskmaster_training_hindi_gujarati_marathi.pkl'
    test_file = '../TaskMasterData/Get_Phones_Combos/1_language/taskmaster_testing_bengali.pkl'

    #build_file = '../TaskMasterData/Get_Phones_Combos/3_lang_variations/taskmaster_training_gujarati_marathi_bengali_hindi_45.pkl'
    #test_file = '../TaskMasterData/Get_Phones_Combos/1_language/taskmaster_testing_bengali.pkl'
    print(build_file)
    print(test_file)
    print('='*20)
    for ngram in range(1,4):
        print('='*30)
        print('')
        print('------------NGRAMS :', ngram, '\n')
        for threshold in range(6):
            
            blockPrint()
            frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
            correct, total, accuracy_per_intent = run_naive_bayes(frequency, word_index, ngram, test_file, all_intents)

            enablePrint()
            print('--THRESHOLD', threshold, ': ACCURACY = ', correct/total)
            print(accuracy_per_intent, '\n')

