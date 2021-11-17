# nowcast_lstm_matlab

MATLAB wrapper for [nowcast_lstm](https://github.com/dhopp1/nowcast_lstm) Python library. [R wrapper](https://github.com/dhopp1/nowcastLSTM) also exists. Long short-term memory neural networks for economic nowcasting. More background in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.

## Poll
To help me better understand the potential userbase and determine whether / which languages to develop future wrappers for, please answer the following poll on your programming language usage.

[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20the%20Python%20library)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20the%20Python%20library/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20R%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I'm%20fine%20with%20R%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20Stata%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20Stata%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20MATLAB%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20MATLAB%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20SAS%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20a%20SAS%20wrapper/vote)
[![](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20an%20SPSS%20wrapper)](https://api.gh-polls.com/poll/01FMEWVADXFWSN5JCWE0M4W9ZP/I%20would%20only%20use%20the%20methodology%20with%20an%20SPSS%20wrapper/vote)

# Installation and setup
**Installing the library in MATLAB**: Simply clone this repo and direct MATLAB to `nowcast_lstm_matlab.m` by putting `addpath('/your_path/nowcast_lstm_matlab/');` at the top of your code. You may have to direct MATLAB to your Python installation the first time by running `pe = pyenv('Version', path_to_python);`
<br><br>
**Installing Python:** Python must be installed on your system for the library to work, however **no Python knowledge is required to use this library**, full functionality can be obtained from MATLAB. Follow [this](https://realpython.com/installing-python/) guide to get Python installed on your system. The pip package manager should also have been installed with Python, if not follow [this](https://www.liquidweb.com/kb/install-pip-windows/) guide for installation on Windows, or [this one](https://pip.pypa.io/en/stable/installing/) for other OSs. [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) is another good option for installing Python.
<br><br>
Once Python is installed, install the following 6 Python libraries, you can use pip for this by running the following from the command line, one after the other. If this isn't working, follow [this](https://packaging.python.org/tutorials/installing-packages/) guide for help. More info on getting pytorch specifically installed is available [here](https://pytorch.org/).

```
# you may have pip3 installed, in which case run "pip3 install..."
pip install dill numpy pandas pmdarima

# pytorch has a little more involved install command, this for windows
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# this for linux
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# then finally
pip install nowcast-lstm
```

**Example**: `nowcast_lstm_matlab_example.zip` contains a MATLAB file with a dataset and more detailed example of usage in MATLAB.

## Background
[LSTM neural networks](https://en.wikipedia.org/wiki/Long_short-term_memory) have been used for nowcasting [before](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), combining the strengths of artificial neural networks with a temporal aspect. However their use in nowcasting economic indicators remains limited, no doubt in part due to the difficulty of obtaining results in existing deep learning frameworks. This library seeks to streamline the process of obtaining results in the hopes of expanding the domains to which LSTM can be applied.

While neural networks are flexible and this framework may be able to get sensible results on levels, the model architecture was developed to nowcast growth rates of economic indicators. As such training inputs should ideally be stationary and seasonally adjusted.

Further explanation of the background problem can be found in [this UNCTAD research paper](https://unctad.org/system/files/official-document/ser-rp-2018d9_en.pdf). Further explanation and results can be found in [this](https://unctad.org/webflyer/economic-nowcasting-long-short-term-memory-artificial-neural-networks-lstm) UNCTAD research paper.

## Quick usage
Given `my_dfs` = a dataframe / table with a date column + monthly data + a quarterly target series to run the model on, usage is as follows:

```MATLAB
addpath('/your_path/nowcast_lstm_matlab/'); % directing MATLAB to this repo
LSTM = nowcast_lstm_matlab; % instantiating the object from which to call the functions
LSTM.initialize_session(); % initializing the accompanying Python session

% data read
my_df = readtable("data.csv");
% making data available in Python session. 2nd argument should be set to string name of dataframe in MATLAB. 3rd argument is name of date column in dataframe.
LSTM.df_matlab_to_python(my_df, "my_df", "date")

% setting LSTM parameters (workaround for no named parameters/default values in MATLAB)
% only these four parameters need to be set, the rest will be set to defaults if not specified
my_params = containers.Map;
my_params('python_model_name') = 'model'; % name of model in Python, used later for predictions and saving trained model
my_params('data') = 'my_df'; % name of dataframe to train model on in Python
my_params('target_variable') = 'target_name';
my_params('n_timesteps') = 12;

% generating full set of LSTM parameters including defaults not specified in `my_params`
params = LSTM.gen_lstm_parameters(my_params);

% instantiating LSTM model. {:} notation is necessary at end of params argument
LSTM.LSTM(params{:})

% training LSTM model, 1st argument is Python model name specified in `my_params('python_model_name')`, second is whether or not to display training epoch losses at the end of training.
LSTM.train("model", false);

% predicting on a trained model, last argument specifies whether to only produce predictions for periods with an actual value present
predictions = LSTM.predict("model", my_df, "my_df", "date", true);

% saving a trained model
LSTM.save_lstm("model", "model.pkl");

% loading a trained model, the first parameter is what the model will be called in Python
LSTM.load("new_model_name", "model.pkl");

```

## LSTM parameters
- `data`: `dataframe / table` of the data to train the model on. Should contain a target column. Any non-numeric columns will be dropped. It should be in the most frequent period of the data. E.g. if I have three monthly variables, two quarterly variables, and a quarterly series, the rows of the dataframe should be months, with the quarterly values appearing every three months (whether Q1 = Jan 1 or Mar 1 depends on the series, but generally the quarterly value should come at the end of the quarter, i.e. Mar 1), with NAs or 0s in between. The same logic applies for yearly variables.
- `target_variable`: a `string`, the name of the target column in the dataframe.
- `n_timesteps`: an `int`, corresponding to the "memory" of the network, i.e. the target value depends on the x past values of the independent variables. For example, if the data is monthly, `n_timesteps=12` means that the estimated target value is based on the previous years' worth of data, 24 is the last two years', etc. This is a hyper parameter that can be evaluated.
- `fill_na_func`: a function used to replace missing values. Pass a Python function, e.g. `np.nanmean`.
- `fill_ragged_edges_func`: a function used to replace missing values at the end of series. Pass `"ARMA"` to use ARMA estimation using `pmdarima.arima.auto_arima`. Options are `"ARMA", "np.nanmean"`, etc.
- `n_models`: `int` of the number of networks to train and predict on. Because neural networks are inherently stochastic, it can be useful to train multiple networks with the same hyper parameters and take the average of their outputs as the model's prediction, to smooth output.
- `train_episodes`: `int` of the number of training episodes/epochs. A short discussion of the topic can be found [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/).
- `batch_size`: `int` of the number of observations per batch. Discussed [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- `decay`: `float` of the rate of decay of the learning rate. Also discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/). Set to `0` for no decay.
- `n_hidden`: `int` of the number of hidden states in the LSTM network. Discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `n_layers`: `int` of the number of LSTM layers to include in the network. Also discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `dropout`: `double` of the proportion of layers to drop in between LSTM layers. Discussed [here](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/).
- `criterion`: `PyTorch loss function`. Discussed [here](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/), list of available options in PyTorch [here](https://pytorch.org/docs/stable/nn.html#loss-functions). Pass as a string, e.g. one of `"torch.nn.L1Loss()", "torch.nn.MSELoss()"`, etc.
- `optimizer`: `PyTorch optimizer`. Discussed [here](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6), list of available options in PyTorch [here](https://pytorch.org/docs/stable/optim.html). Pass as a string, e.g. `"torch.optim.Adam"`.
- `optimizer_parameters`: `String`. Parameters for a particular optimizer, including learning rate. Information [here](https://pytorch.org/docs/stable/optim.html). For instance, to change learning rate (default 1e-2), pass `{"lr":1e-2}`, or weight_decay for L2 regularization, pass `{"lr":1e-2, "weight_decay":0.001}`. Learning rate discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/).

## LSTM outputs
Assuming a model has been instantiated and trained with `LSTM(...); train(...)`, the following functions are available, run `help function` on any of them to find out more about them and their parameters:

- `predict`: to generate predictions on new data
- `save_lstm`: to save a trained model to disk
- `load_lstm`: to load a saved model from disk
- `ragged_preds(python_model_name, pub_lags, lag, data, python_data_name, date_column, start_date, end_date)`: adds artificial missing data then returns a dataframe with date, actuals, and predictions. This is especially useful as a testing mechanism, to generate datasets to see how a trained model would have performed at different synthetic vintages or periods of time in the past. `pub_lags` should be a list of ints (in the same order as the columns of the original data) of length n\_features (i.e. excluding the target variable) dictating the normal publication lag of each of the variables. `lag` is an int of how many periods back we want to simulate being, interpretable as last period relative to target period. E.g. if we are nowcasting June, `lag = -1` will simulate being in May, where May data is published for variables with a publication lag of 0. It will fill with missings values that wouldn't have been available yet according to the publication lag of the variable + the lag parameter. It will fill missings with the same method specified in the `fill_ragged_edges_func` parameter in model instantiation.
- `gen_news(python_model_name, target_period, old_data, new_data, date_column)`: Generates news between one data release to another, adding an element of causal inference to the network. Works by holding out new data column by column, recording differences between this prediction and the prediction on full data, and registering this difference as the new data's contribution to the prediction. Contributions are then scaled to equal the actual observed difference in prediction in the aggregate between the old dataset and the new dataset.
