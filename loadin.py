import pickle

a_file = open("data.pkl", "rb")
output = pickle.load(a_file)

print(len(output))
print(output.keys())

'''for key in output:
    for phone in output[key]['phones']:
        print(phone)

    break'''

