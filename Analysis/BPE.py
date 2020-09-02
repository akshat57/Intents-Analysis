'''
Transform the utterances data into bigram BPE data.
Only need to do it once and you get a BPE dataset.
'''
from get_vocab import load_data, build_ngrams
from naive_bayes import save_data

def most_frequent_bigram(bigrams):
    '''
        bigrams: input is a list of bigrams
        mfb: returns a bigram string which is most frequent
    '''
    count = 0
    mfb = ''


    for i in range(len(bigrams)):
        pair = bigrams[i]
        if pair != mfb and bigrams.count(pair) > count:
            count = bigrams.count(pair)
            mfb = pair

    return mfb, count


def bigram_BPE(utterance, new_symbols):
    '''
        utterance: a list of phones
        new_symbols: any new symbols created while creating 
    '''
    bigram_list = build_ngrams(utterance, 2)
    bigram_list = bigram_list[1:-1]
    mfb, count = most_frequent_bigram(bigram_list)
    utterance = utterance[1:-1]

    if count == 1:
        return utterance, new_symbols

    else:
        utt = []

        i = 0
        while i < len(utterance) - 1:
            bigram = utterance[i] + utterance[i+1]
            if bigram == mfb:
                if mfb not in new_symbols:
                    symbol = str(len(list(new_symbols.keys())))
                    new_symbols[mfb] = symbol

                utt.append(new_symbols[mfb])
                i += 2
            else:
                utt.append(utterance[i])
                i += 1

        if i != len(utterance):
            utt.append(utterance[-1])

        utterance, new_symbols = bigram_BPE(utt, new_symbols)
        return utterance, new_symbols
            

def do_bigram_BPE(data, new_symbols = {}):
    new_utterance = {}
    for key in data:
        new_utterance[key] = []
        for utterance in data[key]:
            utt, new_symbols = bigram_BPE(utterance, new_symbols)
            new_utterance[key].append(utt)

    return new_utterance, new_symbols

if __name__ == '__main__':
    build_file = 'Labels/TaskMaster/taskmaster_training_hindi.pkl'
    test_file = 'Labels/TaskMaster/taskmaster_testing_guj.pkl'

    build_data = load_data(build_file)
    test_data = load_data(test_file)

    new_build_file, new_symbols = do_bigram_BPE(build_data)
    new_test_file, new_symbols = do_bigram_BPE(test_data, new_symbols)

    save_data('Labels/TaskMaster/taskmaster_training_hindi_BPE.pkl', new_build_file)
    save_data('Labels/TaskMaster/taskmaster_testing_hindi_BPE.pkl', new_test_file)
    save_data('Labels/TaskMaster/taskmaster_hindi_BPE_symbols.pkl', new_symbols)
