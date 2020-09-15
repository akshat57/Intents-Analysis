from get_vocab import load_data

save_best_cluster = 'Labels/TaskMaster/best_clusters_gujarati.pkl'
best_clusters = load_data(save_best_cluster)

for kmeans in best_clusters:
    print('')
    print(kmeans, best_clusters[kmeans][0])
    for cluster in best_clusters[kmeans][1]:
        print(best_clusters[kmeans][1][cluster])

    print('='*20)