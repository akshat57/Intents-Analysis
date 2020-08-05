'''Here we train naive bayes on one language and test it on other languages.'''
from get_vocab import load_data
from naive_bayes import build_naive_bayes, run_naive_bayes

languages = ['hindi', 'no_lang', 'marathi', 'bengali', 'punjabi', 'spanish', 'english', 'synthesized_hindi']

for i in range(3):
    frequency, word_index = build_naive_bayes(i+1, 'Labels/intent_labels.pkl')

    for lang in languages:
        print('\n------',lang)
        test_file = 'Labels/intent_' + lang + '_labels.pkl'
        run_naive_bayes(frequency, word_index, i+1, test_file)
    break
