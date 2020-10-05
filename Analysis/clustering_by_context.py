from get_vocab import load_data, get_vocab
from sklearn.cluster import KMeans
import numpy as np
from features_panphon import convert_to_clusters

from naive_bayes import save_data
import operator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from taskmaster_naive import enablePrint, blockPrint
from naive_bayes import build_naive_bayes, run_naive_bayes
import time


def cluster_by_context(N, all_datafile, train_file):
    data = load_data(train_file)
    all_phones, _ = get_vocab(1, all_datafile)

    
    phone_to_index = {}         #create dictionary for phone to index
    index_to_phone = {}         #create dictionary for index to phone
    feature_vectors = {}        #create a dictionary of feature vectors
    for i, phone in enumerate(all_phones):
        phone_to_index[phone] = i
        index_to_phone[i] = phone
        feature_vectors[phone] = [0]*len(all_phones)

    for key in data:
        for utterance in data[key]:
            for i, phone in enumerate(utterance):
                if i != 0:
                    feature_vectors[phone][phone_to_index[utterance[i-1]]] += 1

    #normalize feature vectors
    for vector in feature_vectors:
        normalize = sum(feature_vectors[vector])
        for i in range(len(feature_vectors[vector])):
            feature_vectors[vector][i] /= normalize

    vectors = []
    for i in range(len(all_phones)):
        vectors.append(feature_vectors[index_to_phone[i]])
    
    #Do k-means
    kmeans = KMeans(n_clusters=N)
    kmeans_labels = kmeans.fit_predict(vectors)
    kmeans_trans = kmeans.transform(vectors)

    #Saving clustered data
    clustered = {}
    for index, cluster in enumerate(kmeans_labels):
        if cluster not in clustered:
            clustered[cluster] = [index_to_phone[index]]
        else:
            clustered[cluster].append(index_to_phone[index])

    phone_to_cluster = {}
    for cluster in clustered:
        for phone in clustered[cluster]:
            phone_to_cluster[phone] = str(cluster + 1)

    return phone_to_cluster, clustered, feature_vectors, phone_to_index, index_to_phone

def print_neighbours(clustered, feature_vectors, index_to_phone):
    for cluster in clustered:
        print('cluster:', clustered[cluster])
        for phone in clustered[cluster]:
            index = np.argmax(np.array(feature_vectors[phone]))
            best_indices = np.argsort(-np.array(feature_vectors[phone]))[:5]
            print('--', phone, ':', [index_to_phone[x] for x in best_indices])
        print()

if __name__ == '__main__':
    #Defining constants
    language = 'english'
    lang_threshold = 0.9267
    all_datafile = 'Labels/TaskMaster/data_taskmaster_' + language + '.pkl'
    train_file = 'Labels/TaskMaster/taskmaster_training_' + language + '.pkl'
    test_file = 'Labels/TaskMaster/taskmaster_testing_' + language + '.pkl'
    phone_to_cluster, clustered, feature_vectors, _, index_to_phone = cluster_by_context(15, all_datafile, train_file)
    print_neighbours(clustered, feature_vectors, index_to_phone)

    run = False
    if run:
        save_train = 'Labels/TaskMaster/train_clustered.pkl'
        save_test = 'Labels/TaskMaster/test_clustered.pkl'
        save_best_cluster = 'Labels/TaskMaster/best_clusters_' + language + '.pkl'
        all_intents = ['movie-tickets', 'auto-repair', 'restaurant-table', 'pizza-ordering', 'uber-lyft', 'coffee-ordering']

        accuracy_mean = []
        accuracy_std = []
        best_accuracy = []
        N_kmeans = []
        best_cluster = {}
            
        for N in range(10, 31):
            now = time.time()
            print('='*30)
            print('')
            print('------------NUMBER OF CLUSTERS :', N, '\n')
            accuracy_N = []
            highest_accuracy = 0

            for repitions in range(10):
                phone_to_cluster, clusters, _, _, _ = cluster_by_context(N, all_datafile, train_file)
                train_data_clustered = convert_to_clusters(phone_to_cluster, train_file)
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

