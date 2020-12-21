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
    all_intents = ['1', '2', '3', '4', '5', '6']
    return all_intents


if __name__ == '__main__':

    all_intents = get_domains()

    best_accuracy = []
    best_accuracy_stats = {}
    
    for split in ['1', '2', '3', '4', '5']:
        print('Doing Split ' + split )
        build_file = '../Sinhala_Dataset/datasplit1/sinhala_train_split_' + split + '.pkl'
        test_file = '../Sinhala_Dataset/datasplit1/sinhala_test_split_' + split + '.pkl'
    
        best_acc = 0
        for ngram in range(1, 4):
            #print('='*30)
            #print('')
            #print('------------NGRAMS :', ngram, '\n')
            for threshold in range(6):
            
                blockPrint()
                frequency, word_index = build_naive_bayes(ngram, build_file, threshold)
                correct, total, accuracy_per_intent = run_naive_bayes(frequency, word_index, ngram, test_file, all_intents)

                enablePrint()
                #print('--THRESHOLD', threshold, ': ACCURACY = ', correct/total)
                #print(accuracy_per_intent, '\n')

                current_acc = correct/total
                if current_acc > best_acc:
                    best_acc = current_acc
                    best_accuracy_stats[split] = accuracy_per_intent

        best_accuracy.append(best_acc)

    print('Overall average Accuray:', sum(best_accuracy)/ 5)
    
    for i, split in enumerate(best_accuracy_stats):
        print('='*30)
        print('For split = ' + split, 'Accuracy:', best_accuracy[i])
        print(best_accuracy_stats[split])


            
            
