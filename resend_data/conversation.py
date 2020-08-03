from conversation_function import detect_intent_audio
import sys
import os

project_id = 'hinglish-banker-inwvsx'
#session_id = '1'
language_code = 'hi'

audio_file_path = sys.argv[1]
sample_rate = int(sys.argv[2])
session_id = sys.argv[3]
response, query, intent, name = detect_intent_audio(project_id, session_id, audio_file_path, language_code, sample_rate)

info_file = name + '_info'
query_file = name + '_query'
response_file = name + '_response'
intent_file = name + '_intent'

file = open( response_file, 'w')
file.write(response)
file.close()

file = open( query_file, 'w')
file.write(query)
file.close()

file = open( intent_file, 'w')
file.write(intent)
file.close()

file = open( info_file, 'w')
file.write('Audio File Name : ' +  name + '\n')
#file.write('Query File : ' + query_file + '\n')
#file.write('Response File : ' + response_file + '\n')

#file.write('\n')
file.write('Query : ' +  query + '\n')
file.write('Response : ' + response + '\n' )
file.write('Intent : ' +  intent)
file.close()

print('conversation.....', os.getcwd())
