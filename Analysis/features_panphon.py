from get_vocab import load_data, get_vocab
from naive_bayes import save_data
import panphon
import numpy as np
import operator

ft = panphon.FeatureTable()

build_file = 'Labels/TaskMaster/taskmaster_training_hindi.pkl'
test_file = 'Labels/TaskMaster/taskmaster_testing_hindi.pkl'

build_data = load_data(build_file)
test_data = load_data(test_file)


vocab, _ = get_vocab(1, build_file)
vectors = {}
for ipa in vocab:
    vectors[ipa] = np.array(ft.fts(ipa).numeric())

###saving data
save_data('panphon_data.pkl', vectors)


closest = {}
for ipa in vectors:
    temp = {}
    for i,l in enumerate(vectors):
        distance = np.linalg.norm(np.abs(vectors[ipa] - vectors[l]))
        temp[l] = distance

    closest[ipa] = sorted(temp.items(), key=operator.itemgetter(1))


print(ipa, closest[ipa])


for key in build_data:
    for utterance in build_data[key]:
        for ipa in utterance:
            a = ft.fts(ipa).numeric()
            #print(ipa, len(ipa), a)

        '''utt = ''.join(utterance)
        print(utt, len(utt))
        a = ft.word_to_vector_list(utt, numeric=True)
        print(len(a))'''
        break
    break
