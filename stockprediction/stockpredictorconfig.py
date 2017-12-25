# -*- coding: utf-8 -*-

import sys

# flag to check if python3 is used
PY3 = sys.version_info[0] == 3


class StockpredictorConfig(object):
    """
    Config class.
    Used to configure the stockpredictor model.
    """

    RNN_MODES = {
        "BASIC": "basic",
        "CUDNN": "cudnn",
        "BLOCK": "block"
    }

    # init_scale = 0.04
    learning_rate = 1.0     # The learning rate
    # max_grad_norm = 10
    # num_layers = 2
    # num_steps = 35
    hidden_size = 1        # The number of hidden layers within one ltsm cell
    # max_epoch = 14
    # max_max_epoch = 55
    training_steps = 10000
    # keep_prob = 0.35
    # lr_decay = 1 / 1.15
    batch_size = 5
    time_steps = 30         # 30 days
    values = 15             # stock value, googletrends counter
    num_classes = 2
    # vocab_size = 10000
    rnn_mode = RNN_MODES["BASIC"]
