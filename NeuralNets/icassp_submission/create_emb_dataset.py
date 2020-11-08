import sys
sys.path.insert(1, '/Users/manjugupta/Desktop/CMU_Courses/Intents/getting_intents/Analysis')

from get_vocab import load_data, get_vocab
from naive_bayes import save_data

def create_training_data(all_phones, data):
    #contains training data for embeddings for CBOW
    embedding_data = {}         
    for phone in all_phones:
        embedding_data[phone] = []

    for intent in data:
        for utterance in data[intent]:
            for c in range(context_size):
                utterance.insert(0, 'unk')
                utterance.append('unk')
            for i, phone in enumerate(utterance):
                if phone!= 'unk':
                    window = utterance[i-context_size:i] + utterance[i +1: i + context_size + 1]
                    embedding_data[phone].append(window)

    return embedding_data

if __name__ == '__main__':
    language = 'english'
    context_size = 2
    train_file = '../Analysis/Labels/TaskMaster/taskmaster_training_' + language + '.pkl'
    
    data = load_data(train_file)
    all_phones, _ = get_vocab(1, train_file)

    embedding_data = create_training_data(all_phones, data)
    save_data('embedding_data_' + language + '_' + str(context_size) + '.pkl', embedding_data)