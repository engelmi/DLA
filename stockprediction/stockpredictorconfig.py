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

    learning_rate = 1.0
    hidden_size = 100
    num_epochs = 2000
    cross_validation_k = 10
    batch_size = 5
    time_steps = 30
    values = 15
    num_classes = 2
    rnn_mode = RNN_MODES["BASIC"]
