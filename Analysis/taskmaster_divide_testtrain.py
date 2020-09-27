from get_vocab import load_data
from naive_bayes import save_data

language = 'marathi'
taskmaster_data = load_data('Labels/TaskMaster/data_taskmaster_' + language + '.pkl')
labels = list(taskmaster_data.keys())

test_data = {}
training_data = {}
size = 50

#shuffle the list for future iterations
for label in labels:
    test_data[label] = taskmaster_data[label][0:size]
    training_data[label] = taskmaster_data[label][size:]

save_data('Labels/TaskMaster/taskmaster_training_' + language + '.pkl', training_data)
save_data('Labels/TaskMaster/taskmaster_testing_' + language + '.pkl', test_data)
