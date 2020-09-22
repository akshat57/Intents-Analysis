from get_vocab import load_data
from naive_bayes import save_data

taskmaster_data = load_data('Labels/TaskMaster/data_taskmaster_hindi_2voices.pkl')
labels = list(taskmaster_data.keys())

test_data = {}
training_data = {}
size = 50

#shuffle the list for future iterations
for label in labels:
    test_data[label] = taskmaster_data[label][0:size]
    training_data[label] = taskmaster_data[label][size:]

save_data('Labels/TaskMaster/taskmaster_training_hindi_2voices.pkl', training_data)
save_data('Labels/TaskMaster/taskmaster_testing_hindi_2voices.pkl', test_data)
