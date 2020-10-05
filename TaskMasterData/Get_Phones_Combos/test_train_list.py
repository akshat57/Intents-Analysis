#creates a test-train split by creating list of indices in train and test set
import sys
import random
sys.path.insert(1, '/Users/manjugupta/Desktop/CMU_Courses/Intents/getting_intents/Analysis')

from get_vocab import load_data
from naive_bayes import save_data

index_to_intents = load_data('index_to_intents.pkl')
intents_to_index = {}

for index in index_to_intents:
    if index_to_intents[index] not in intents_to_index:
        intents_to_index[index_to_intents[index]] = [index]
    else:
        intents_to_index[index_to_intents[index]].append(index)

test_indices = {}
train_indices = {}
for intent in intents_to_index:
    random.shuffle(intents_to_index[intent])
    test_indices[intent] =  intents_to_index[intent][:50]
    train_indices[intent] = intents_to_index[intent][50:]

save_data('test_indices.pkl', test_indices)
save_data('train_indices.pkl', train_indices)