from get_vocab import get_vocab
from naive_bayes import save_data
import panphon
import numpy as np

ft = panphon.FeatureTable()

data_file = 'Labels/TaskMaster/data_taskmaster_gujarati.pkl'

vocab, _ = get_vocab(1, data_file)
vectors = {}
for ipa in vocab:
    vectors[ipa] = np.array(ft.fts(ipa).numeric())

###saving data
save_data('panphon_features_gujarati.pkl', vectors)
