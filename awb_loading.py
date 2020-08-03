import pickle

a_file = open("data.pkl", "rb")
output = pickle.load(a_file)
a_file.close()

print(output)
