import pickle
import os
import operator
import argparse

def summarize(lang = 'hindi'):
    data_file = '../Phone_Data/data_' + lang + '.pkl'
    out_file = 'Labels/' + lang + '_labels.txt'
    
    a_file = open( data_file, "rb")
    output = pickle.load(a_file)
    a_file.close()

    transcript_to_phones = {}
    print('')
    for key in output:
        phones = ''
        for phone in output[key]['phones']:
            phones += phone.decode('utf-8')

        #os.system('cp ../' + output[key]['location'] + key + ' ../audio_files/')
        transcript_to_phones[key] = [output[key]['location'], output[key]['intent'], output[key]['text'], phones]

    sorted_ttp = sorted(transcript_to_phones.items(), key=operator.itemgetter(0))

    output = []
    for audio in sorted_ttp:
        row = audio[0] + ',' + audio[1][0] + ',' + audio[1][1] + ',' + audio[1][2] + ',' + audio[1][3]
        output.append(row)
        print(audio[0], ',', audio[1][0], ',', audio[1][1], ',', audio[1][2], ',', audio[1][3])

    print('')

    #Storing data in text file as need to edit data by hand.
    with open(out_file, 'w') as filehandle:
        for sample in output:
            filehandle.write('%s\n' % sample)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whatever')
    parser.add_argument('--lang', dest='lang', default='hindi', nargs='?')
    args = parser.parse_args()

    summarize(args.lang)


