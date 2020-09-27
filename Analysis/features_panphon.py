from get_vocab import load_data
from naive_bayes import save_data
import numpy as np
import operator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from taskmaster_naive import enablePrint, blockPrint
from naive_bayes import build_naive_bayes, run_naive_bayes
import time

def get_clusters(N = 10, feature_file = 'Labels/panphon_features_hindi.pkl'):
    feature_vectors = load_data(feature_file)

    vectors = []
    ipas = []
    for ipa in feature_vectors:
        vectors.append(list(feature_vectors[ipa]))
        ipas.append(ipa)

    #Do k-means
    kmeans = KMeans(n_clusters=N)
    kmeans_labels = kmeans.fit_predict(vectors)
    kmeans_trans = kmeans.transform(vectors)

    #Saving clustered data
    clustered = {}
    for index, cluster in enumerate(kmeans_labels):
        if cluster not in clustered:
            clustered[cluster] = [ipas[index]]
        else:
            clustered[cluster].append(ipas[index])

    phone_to_cluster = {}
    for cluster in clustered:
        for phone in clustered[cluster]:
            phone_to_cluster[phone] = str(cluster + 1)

    return phone_to_cluster, clustered


def convert_to_clusters(phone_to_cluster, file_name):
    data = load_data(file_name)

    data_clustered = {}
    for key in data:
        data_clustered[key] = []
        for utterance in data[key]:
            utterance_to_cluster = []
            for phone in utterance:
                utterance_to_cluster.append(phone_to_cluster[phone])
            data_clustered[key].append(utterance_to_cluster)

    return data_clustered



if __name__ == '__main__':
    #Defining constants
    language = 'english'
    lang_threshold = 0.9267
    feature_file = 'Labels/panphon_features_' + language + '.pkl'
    build_file = 'Labels/TaskMaster/taskmaster_training_' + language + '.pkl'
    test_file = 'Labels/TaskMaster/taskmaster_testing_' + language + '.pkl'
    save_train = 'Labels/TaskMaster/train_clustered.pkl'
    save_test = 'Labels/TaskMaster/test_clustered.pkl'
    save_best_cluster = 'Labels/TaskMaster/best_clusters_' + language + '.pkl'
    all_intents = ['movie-tickets', 'auto-repair', 'restaurant-table', 'pizza-ordering', 'uber-lyft', 'coffee-ordering']

    accuracy_mean = []
    accuracy_std = []
    best_accuracy = []
    N_kmeans = []
    best_cluster = {}
        
    for N in range(5, 31):
        now = time.time()
        print('='*30)
        print('')
        print('------------NUMBER OF CLUSTERS :', N, '\n')
        accuracy_N = []
        highest_accuracy = 0

        for repitions in range(10):
            phone_to_cluster, clusters = get_clusters(N, feature_file)
            train_data_clustered = convert_to_clusters(phone_to_cluster, build_file)
            test_data_clustered = convert_to_clusters(phone_to_cluster, test_file)

            save_data(save_train, train_data_clustered)
            save_data(save_test, test_data_clustered)

            for ngram in range(3,4):

                for threshold in range(1):
                    
                    blockPrint()
                    frequency, word_index = build_naive_bayes(ngram, save_train, threshold)
                    correct, total, accuracy_per_intent = run_naive_bayes(frequency, word_index, ngram, save_test, all_intents)

                    enablePrint()
                    print('--NGRAM', ngram, ': ACCURACY = ', correct/total)
                    #print(accuracy_per_intent, '\n')
                    accuracy_N.append(correct/total)

            if correct/total > highest_accuracy:
                highest_accuracy = correct/total
                best_cluster[N] = [correct/total, clusters]

        accuracy_mean.append(np.mean(np.array(accuracy_N)))
        accuracy_std.append(np.std(np.array(accuracy_N)))
        best_accuracy.append(max(accuracy_N))
        N_kmeans.append(N)
        print('Time Taken:', time.time()-now, ' -- Mean Accuracy:', accuracy_mean[-1], '-- std:', accuracy_std[-1])

    save_data(save_best_cluster ,best_cluster)
    plt.errorbar(N_kmeans, accuracy_mean, accuracy_std, fmt='-o')
    plt.plot(N_kmeans, best_accuracy, 'r')
    plt.plot(N_kmeans, np.full(len(N_kmeans), lang_threshold), '-k')
    plt.legend(['Best Accuracy Clustering', 'w/o clustering', 'Clustered Accuracy (Mean/Std)'])
    plt.show()
