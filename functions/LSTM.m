function f = LSTM(data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters)
% LSTM Instantiate an LSTM model. Pass output of gen_lstm_parameters function, e.g. x = gen_lstm_parameters(my_map); LSTM(x{:})
% Arguments:
% -data: MATLAB dataframe of training data
% -target_variable: String, name of the target variable column
% -n_timesteps: Int, how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
% -fill_na_func: String, function to replace within-series NAs. Given a column, the function should return a scalar. Enter a Python function name that returns a scalar as a string, e.g. 'np.nanmean' for the mean of the series.
% -fill_ragged_edges_func: String, function to replace NAs in ragged edges (data missing at end of series). Pass "ARMA" for ARMA filling. Not ARMA filling will be significantly slower as models have to be estimated for each variable to fill ragged edges.
% -n_models: Int, number of models to train and take the average of for more robust estimates
% -train_episodes: Int, number of epochs/episodes to train the model
% -batch_size: Int, number of observations per training batch
% -decay: Double, learning rate decay
% -n_hidden: Int, number of hidden states in the network
% -n_layers: Int, number of LSTM layers in the network
% -dropout: Double, dropout rate between the LSTM layers
% -criterion: String, torch loss criterion, defaults to MAE. Of form 'torch.nn.L1Loss()'
% -optimizer: String, torch optimizer, defaults to Adam. Of form 'torch.optim.Adam '
% -optimizer_parameters: String, list of parameters for optimizer, including learning rate. Pass string in form of Python Dictionary, e.g. '{"lr": 1e-2, "weight_decay": 1e-4}'
    model_name = char(randi([33 126],1,40)); % randomly generated name for Python variable, stored in output of this function
    model_name = model_name(isstrprop(model_name,'alpha'));

    % converting numeric arguments to strings to pass to pyrun
    n_timesteps = string(n_timesteps);
    n_models = string(n_models);
    train_episodes = string(train_episodes);
    batch_size = string(batch_size);
    decay= string(decay);
    n_hidden = string(n_hidden);
    n_layers = string(n_layers);
    dropout = string(dropout);
    if fill_ragged_edges_func == "ARMA"
        fill_ragged_edges_func = "'ARMA'";
    end

    pyrun(sprintf("%s = LSTM(data=%s, target_variable='%s', n_timesteps=%s, fill_na_func=%s, fill_ragged_edges_func=%s, n_models=%s, train_episodes=%s, batch_size=%s, decay=%s, n_hidden=%s, n_layers=%s, dropout=%s, criterion=%s, optimizer=%s, optimizer_parameters=%s)", model_name, data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters))

    f = model_name;
end