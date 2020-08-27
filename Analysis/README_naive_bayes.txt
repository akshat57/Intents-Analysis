We have functions spread accross various files that are used to run naive bayes' classifier.


1. get_vocab.py : This is used to get the N-gram vocabulary of all the utterances in the dataset.
    This also has three functions that are used in downstream naive-bayes calculation.

    - get_vocab : 
        INPUTS:
            N : N in ngrams 
            filename : .pkl data file as described below.

        DESCRIPTION:
            This function takes input the value of N in the N-grams to be constructed and the .pkl data file.
            The .pkl data file is a dictionary with intents as keys and all utterances of that intent as values.

    - build_ngrams : 
        INPUTS:
            dialogue: Dialogue as a list of phones in the utterance:
            N : N in ngrams

        DESCRIPTION:
            This function takes input an utterance and N. 
            It outputs a list of ngrams making up that utterance. Each Ngram is a string


    - load_data: 
        INPUTS:
            filename: Any .pkl file.
        
        DESCRIPTION:
            This function loads data from any .pkl file. Returns the data stored in the .pkl file.



2. get_frequency.py : This is used to create unique Ngrams and their frequencies. 
    This also contains 2 functions that are used in downstream calculations.

    - word_to_index:
        INPUTS:
            vocab : is a list of tuples containing the ngram and frequency of each ngram. It is a sorted list.

        DESCRIPTION:
            This function converts words to indices which. Output is a word to index dictionary. 
            These indices are used for downstream calculations.

    - get_frequency: 
        INPUTS:
            N: ngram
            filename: .pkl file as in get_vocab

        DESCRIPTION:
            This function creates a frequency distribution for all intents. The frequencies are not normalized.
            The output is a dictionary of dictionaries, where intents are primary the keys. For each intent, there is a mapping between word utterances and its number of occurances.



3. naive_bayes.py

    -build_naive_bayes:
        INPUTS:
            N: ngram
            filename : This is the file for training set. This has the same filename format as mentioned in functions above.
            threshold: This decides the kind of smoothing we use. For threshold = 0, we use add-1 smoothing. For threshold >0, we use leave k out smoothing.

        DESCRIPTION:
            This function builds the naive bayes classifier on the training set. The outputs of this function are then used to test the naive bayes on the test set.

    -run_naive_bayes: should be called test_naive_bayes. This tests the naive bayes built in build_naive_bayes
        INPUTS:
            frequency: output from build_naive_bayes
            word_index: output from build_naive_bayes
            N: ngram
            filename: Test file name 
            all_intents: A list of all intents. Its good to provide all intents list as all the intents might not be present in the test file.


        DESCRIPTION:
            Tests the naive bayes built on the test set provided.

