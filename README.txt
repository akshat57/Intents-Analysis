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

	- 
