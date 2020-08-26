We have the following files : 

	- summarize_data_from_awbpc.py : 
		This shows a summary of data that comes from awbpc after pre processing for intent recognition.
		You can re-direct the ouput of this code to a file and change the data manually.
		Run the code using : python3 summarize_data_from_awbpc.py > intent_recognition_labels.txt   

	- pre-processing_awbpc.py :
		This pre-processes data from the data.pkl that comes from awbpc. 

	- pre-processing_mypc.py :
		This pre-processes data from intent_recognition_labels.txt created above.

	- get_vocab.py :
		This is used to get the N-gram vocabulary of all the utterances.

	- get_frequency.py :
		This is used to get the frequency of occurences of N-gram phones.

	- naive_bayes.py :
		This runs naive bayes for the N-gram case. Also does cross validation. 


The process:
1. Run 'summarize_data_from_awbpc.py' to get a summary of data. This is used to build both train and test data.
	RUN DEFAULT : python3 summarize_data_from_awbpc.py
	RUN WITH FLAGS : python3 summarize_data_from_awbpc.py --lang marathi

2. Run DEFAULT 'pre_processing_mypc.py' with the file you want to use as training data. Use that to build 'intent_labels.pkl'.
	To pre-process other language data, which is not training data,
	RUN : python3 pre_processing_mypc.py --lang marathi --flag Test

SYNTHESIZED data
1. RUN synthesized_summarize.py to convert synthesize data file into a labels.txt file.
2. RUN synthesized_stats.py to see similarity index values for two given speech samples.
3. 




SYNTHESIZED FEMALE 1 IS ARCHANA
SYNTHESIZED FEMALE 2 IS IITF
SYNTHESIZED IS SYNTHESIZED MALE VOICE