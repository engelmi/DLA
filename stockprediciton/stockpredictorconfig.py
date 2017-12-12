# -*- coding: utf-8 -*-

import sys

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3

"""
Config class.
Used to configure the stockpredictor model. 
"""
class StockpredictorConfig(object):
	
	RNN_MODES = {
		"BASIC" : "basic", 
		"CUDNN" : "cudnn", 
		"BLOCK" : "block"
	}
	
	#init_scale = 0.04
	learning_rate = 1.0
	#max_grad_norm = 10
	#num_layers = 2
	#num_steps = 35
	hidden_size = 200
	#max_epoch = 14
	#max_max_epoch = 55
	#keep_prob = 0.35
	#lr_decay = 1 / 1.15
	batch_size = 30		# 30 days at once
	time_steps = 16		# stock value, label, googletrends counter
	#vocab_size = 10000
	rnn_mode = RNN_MODES["BASIC"]
	
	
