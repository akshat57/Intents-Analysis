from pre_processing_mypc import *

languages = ['hindi', 'no_lang', 'marathi', 'bengali', 'punjabi', 'spanish', 'english']

unique_phones = []
for lang in languages:
    print('###', lang, '###')
    unique_phones.append(preprocess(lang, 'Test'))
    print('-'*20)

print('Number of Unique Hindi Phones:', len(unique_phones[0]))
print('Intersection with:')
for i in range(1,len(languages)):
    intersection = len(unique_phones[0].intersection(unique_phones[i]))
    print('--', languages[i], 'Total :',  len(unique_phones[i]), '-- Intersection: ', intersection)
