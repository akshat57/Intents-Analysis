from get_vocab import get_vocab
from naive_bayes import save_data
import panphon
import numpy as np

ft = panphon.FeatureTable()

language = 'marathi'

data_file = 'Labels/TaskMaster/data_taskmaster_' + language + '.pkl'

vocab, _ = get_vocab(1, data_file)
vectors = {}
for ipa in vocab:
    vectors[ipa] = np.array(ft.fts(ipa).numeric())

###saving data
save_data('Labels/TaskMaster/panphon_features_' + language + '.pkl', vectors)
