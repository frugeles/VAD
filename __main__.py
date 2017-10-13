import glob
import os
import csv
import numpy as np
import tensorflow as tf
import time
import logging

from datetime import timedelta
from vad_model_v4 import VADModel
from feature_extractor import FeatureExtractor
from dataset_utils import TRStoCSV
from dataset_utils import normalize_wav
from itertools import islice

def train_model():
	""" Model Training code snippet
	"""
	
	start_time = time.time()
	with tf.Graph().as_default():
		model = VADModel.build(param_dir)
		with tf.Session() as session:
			cost_history, training_accuracy, training_perplexity = model.train(session, X_train, Y_train)
	print("Total training time %s" % timedelta(seconds=(time.time() - start_time)))


def evaluate_model():
	""" Model Evaluation code snippet
	"""
	
	with tf.Graph().as_default():
		with tf.Session() as session:
			model = VADModel.restore(session, param_dir)
			accuracy, perplexity = model.evaluate(session, X_test, Y_test)

	print("Perplexity=", "{:.4f}".format(evaluation_perplexity),
		", Accuracy= ", "{:.5f}".format(evaluation_accuracy))
	
def configure_logging(log_filename):
	logger = logging.getLogger("rnnlogger")
	logger.setLevel(logging.DEBUG)
	# Format for our loglines
	formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	# Setup console logging
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	# Setup file logging as well
	fh = logging.FileHandler(log_filename)
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	return logger



def main():
	print("Shall we start ??...")
	
	print("\nReading CHIME dataset ...")

	X_chime=[]

	# with open('dataset/X_CHIME_dummy_withhot.csv', 'r') as f:
	with open('dataset/X_CHIME_dummy_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)
        
	for l in data:
		X_chime.append([float(i) for i in l])

	Y_chime=[]
		
	# with open('dataset/Y_CHIME_dummy_withhot.csv', 'r') as f:
	with open('dataset/Y_CHIME_dummy_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		Y_chime.append([int(float(i)) for i in l])

	X_chime=np.asarray(X_chime)
	Y_chime=np.asarray(Y_chime)
	print("CHIME :\n  | Features : ",X_chime.shape,"\n  | Labels : ",Y_chime.shape)

	print("\nReading Transcript dataset ...")

	X_transcript=[]

	# with open('dataset/X_Transcript_withhot.csv', 'r') as f:
	with open('dataset/X_Transcript_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)
        
	for l in data:
		X_transcript.append([float(i) for i in l])

	Y_transcript=[]
		
	# with open('dataset/Y_Transcript_withhot.csv', 'r') as f:
	with open('dataset/Y_Transcript_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		Y_transcript.append([int(float(i)) for i in l])

	X_transcript=np.asarray(X_transcript)
	Y_transcript=np.asarray(Y_transcript)
	print("Transcript :\n  | Features : ",X_transcript.shape,"\n  | Labels : ",Y_transcript.shape)	
		
	print("Splitting dataset on training/test")

	stop = len(X_chime)
	# stop = int(len(X_chime)*0.3)
	split = int(stop*0.7)

	stop_tr = len(X_transcript)
	split_tr = int(stop_tr*0.7)

	# X_chime_temp = X_chime[:stop,:]
	# Y_chime_temp = Y_chime[:stop]
	# Y_chime_temp = Y_chime[:stop,:]

	# dataset_dir='dataset'	

	# with open(os.path.join(dataset_dir,'X_CHIME_dummy_withhot.csv'), 'w') as a:
	# 	wxtest = csv.writer(a)
	# 	wxtest.writerows(X_chime_temp)


	# print("Dummy CHiME features saved to ",os.path.join(dataset_dir,'X_CHIME_dummy_withhot.csv'))

	# with open(os.path.join(dataset_dir,'Y_CHIME_dummy_withhot.csv'), 'w') as b:
	# 	wytest = csv.writer(b)
	# 	wytest.writerows(Y_chime_temp)
	# 	# for e in Y_chime_temp:
	# 	# 	wytest.writerow(e)

	# print("Dummy CHiME labels saved to ",os.path.join(dataset_dir,'Y_CHIME_dummy_withhot.csv'))

	X_train, X_test = np.concatenate((X_chime[:split,:], X_transcript[:split_tr,:])), np.concatenate((X_chime[split:,:], X_transcript[split_tr:,:]))
	# X_train, X_test = X_chime_temp[:split,:], X_chime_temp[split:,:]

	# Y_train, Y_test = Y_chime[:split,:], Y_chime[split:,:]
	Y_train, Y_test = np.concatenate((Y_chime[:split], Y_transcript[:split_tr])), np.concatenate((Y_chime[split:], Y_transcript[split_tr:]))
	# Y_train, Y_test = np.concatenate((Y_chime[:split,:], Y_transcript[:split_tr,:])), np.concatenate((Y_chime[split:,:], Y_transcript[split_tr:,:]))

	print("Training : \n  |  Features : ",X_train.shape,"\n  |  Labels : ",Y_train.shape)
	print("Test : \n  |  Features : ",X_test.shape,"\n  |  Labels : ",Y_test.shape,"\n\n")

	label_speech=0
	label_nonspeech=0

	for row in Y_train:
		if row[0] == 1:
			label_speech=label_speech+1
		if row[0] == 0:
			label_nonspeech=label_nonspeech+1

	print("Training : \n  |  Speech : ",label_speech,"\n  |  Non-Speech : ",label_nonspeech)

	label_speech=0
	label_nonspeech=0

	for row in Y_test:
		if row[0] == 1:
			label_speech=label_speech+1
		if row[0] == 0:
			label_nonspeech=label_nonspeech+1

	print("Test : \n  |  Speech : ",label_speech,"\n  |  Non-Speech : ",label_nonspeech)

	
	"""
	print("\nReading SWEETHOME Multimodale dataset ...")
	stop = 2559663
	"""

	""" Training code snippet
	"""
	
	param_dir='parameters'

	start_time = time.time()

	with tf.Graph().as_default():
		model = VADModel.build(param_dir)
		with tf.Session() as session:
			cost_history, training_accuracy, _ , _ = model.train(session, X_train, Y_train)
	print("Total training time %s" % timedelta(seconds=(time.time() - start_time)))
	
	
	""" Evaluation code snippet
	"""
	
	with tf.Graph().as_default():
		with tf.Session() as session:
			model = VADModel.restore(session, param_dir)
			evaluation_accuracy, _ , _ = model.evaluate(session, X_test, Y_test)

	print("Accuracy= ", "{:.5f}".format(evaluation_accuracy))




	""" Test code snippet
	"""

	# print("\nReading SWEETHOME Parole dataset ...")

	# X_sh_parole=[]

	# # with open('dataset/X_SWEETHOME_Parole_withhot.csv', 'r') as f:
	# with open('dataset/X_SWEETHOME_Parole_nohot.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	#next(reader, None)  # skip the headers
	# 	data = list(reader)
        
	# for l in data:
	# 	X_sh_parole.append([float(i) for i in l])

	# Y_sh_parole=[]
		
	# # with open('dataset/Y_SWEETHOME_Parole_withhot.csv', 'r') as f:
	# with open('dataset/Y_SWEETHOME_Parole_nohot.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	#next(reader, None)  # skip the headers
	# 	data = list(reader)

	# for l in data:
	# 	Y_sh_parole.append([int(float(i)) for i in l])

	# X_sh_parole=np.asarray(X_sh_parole)
	# Y_sh_parole=np.asarray(Y_sh_parole)

	# print("\nReading SWEETHOME Multimodal dataset ...")

	# X_sh_multimodal=[]

	# # with open('dataset/X_SWEETHOME_Multimodal_withhot.csv', 'r') as f:
	# with open('dataset/X_SWEETHOME_Multimodal_nohot.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	#next(reader, None)  # skip the headers
	# 	data = list(reader)
        
	# for l in data:
	# 	X_sh_multimodal.append([float(i) for i in l])

	# Y_sh_multimodal=[]
		
	# # with open('dataset/Y_SWEETHOME_Multimodal_withhot.csv', 'r') as f:
	# with open('dataset/Y_SWEETHOME_Multimodal_nohot.csv', 'r') as f:
	# 	reader = csv.reader(f)
	# 	#next(reader, None)  # skip the headers
	# 	data = list(reader)

	# for l in data:
	# 	Y_sh_multimodal.append([int(float(i)) for i in l])

	# X_sh_multimodal=np.asarray(X_sh_multimodal)
	# Y_sh_multimodal=np.asarray(Y_sh_multimodal)

	# stop_sh = 2559664 

	# X_final_test = np.concatenate((X_sh_parole, X_sh_multimodal[:stop_sh,:]))

	# # Y_final_test = np.concatenate((Y_sh_parole, Y_sh_multimodal[:stop_sh,:]))
	# Y_final_test = np.concatenate((Y_sh_parole, Y_sh_multimodal[:stop_sh]))

	# label_speech=0
	# label_nonspeech=0

	# for row in Y_final_test:
	# 	if row[0] == 1:
	# 		label_speech=label_speech+1
	# 	if row[0] == 0:
	# 		label_nonspeech=label_nonspeech+1

	# print("SwetHome : \n  |  Speech : ",label_speech,"\n  |  Non-Speech : ",label_nonspeech)

	# dataset_dir='dataset'

	# with open(os.path.join(dataset_dir,'X_SWEETHOME_nohot.csv'), 'w') as a:
	# 	wxtest = csv.writer(a)
	# 	wxtest.writerows(X_final_test)

	# with open(os.path.join(dataset_dir,'Y_SWEETHOME_nohot.csv'), 'w') as b:
	# 	wytest = csv.writer(b)
	# 	for e in Y_final_test:
	# 		wytest.writerow([e])

	# with open(os.path.join(dataset_dir,'X_SWEETHOME_withhot.csv'), 'w') as a:
	# 	wxtest = csv.writer(a)
	# 	wxtest.writerows(X_final_test)

	# with open(os.path.join(dataset_dir,'Y_SWEETHOME_withhot.csv'), 'w') as b:
	# 	wytest = csv.writer(b)
	# 	wytest.writerows(Y_final_test)

	print("\nReading SWEETHOME dataset ...")

	X_sh=[]

	# with open('dataset/X_SWEETHOME_withhot.csv', 'r') as f:
	with open('dataset/X_SWEETHOME_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)
        
	for l in data:
		X_sh.append([float(i) for i in l])

	Y_sh=[]
		
	# with open('dataset/Y_SWEETHOME_withhot.csv', 'r') as f:
	with open('dataset/Y_SWEETHOME_nohot.csv', 'r') as f:
		reader = csv.reader(f)
		#next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		Y_sh.append([int(float(i)) for i in l])

	X_sh=np.asarray(X_sh)
	Y_sh=np.asarray(Y_sh)

	label_speech=0
	label_nonspeech=0

	for row in Y_sh:
		if row[0] == 1:
			label_speech=label_speech+1
		if row[0] == 0:
			label_nonspeech=label_nonspeech+1

	print("SweetHome : \n  |  Speech : ",label_speech,"\n  |  Non-Speech : ",label_nonspeech)


	with tf.Graph().as_default():
		with tf.Session() as session:
			model = VADModel.restore(session, param_dir)
			# evaluation_accuracy, _ , _ = model.evaluate(session, X_final_test, Y_final_test)
			evaluation_accuracy, _ , _ = model.evaluate(session, X_sh, Y_sh)

	print("Accuracy= ", "{:.5f}".format(evaluation_accuracy))
	


	print("And that's it for today ladies and gentlemen!...")

if __name__ == "__main__":
	#logger = configure_logging()
	main()