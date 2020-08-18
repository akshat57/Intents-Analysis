a_file = open( 'intent_text.txt', "r")
text = a_file.readlines()
a_file.close()

num_char = 0
for utterance in text:
    num_char += len(utterance.split('|')[1])

print('Total Number of Characters:', num_char)
print('Avg Number of Characters per utterance:', num_char/len(text))
