function f = initialize_session()
% initialize_session Initialize accompanying Python session. Run at the beginning of every session.
    pyrun("import os")
    pyrun("from nowcast_lstm.LSTM import LSTM")
    pyrun("import pandas as pd")
    pyrun("import numpy as np")
    pyrun("import dill")
    pyrun("import torch")
end