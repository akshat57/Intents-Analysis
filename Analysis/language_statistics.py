from get_vocab import get_vocab, load_data
import seaborn as sns
import matplotlib.pyplot as plt

languages = ['english', 'hindi', 'gujarati', 'bengali', 'marathi']

all_phones = {}
for lang in languages:
    datafile = 'Labels/TaskMaster/data_taskmaster_' + lang + '.pkl'
    all_phones[lang], _ = get_vocab(1, datafile)

#printing common phones across languages
for lang in languages:
    print(lang, len(all_phones[lang]))
    for second_lang in languages:
        if second_lang != lang:
            print('---', second_lang, len(all_phones[lang].intersection(all_phones[second_lang]))/len(all_phones[second_lang]))
    print()

#printing length of sentences
sent_sizes = {}
max_length = {}
for lang in languages:
    datafile = 'Labels/TaskMaster/data_taskmaster_' + lang + '.pkl'  
    data = load_data(datafile)
    sent_sizes[lang] = []
    for key in data:
        for utterance in data[key]:
            sent_sizes[lang].append(len(utterance))
    
    
    print(lang, '-- Max Length :', max(sent_sizes[lang]))
    #plot sentence lengths
    sns.distplot(sent_sizes[lang])

plt.legend(languages)
plt.show()




'''unique_phones = []
for lang in languages:
    print('###', lang, '###')
    unique_phones.append(preprocess(lang, 'Test'))
    print('-'*20)

print('Number of Unique Hindi Phones:', len(unique_phones[0]))
print('Intersection with:')
for i in range(1,len(languages)):
    intersection = len(unique_phones[0].intersection(unique_phones[i]))
    print('--', languages[i], 'Total :',  len(unique_phones[i]), '-- Intersection: ', intersection)
'''