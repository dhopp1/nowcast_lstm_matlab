classdef nowcast_lstm_matlab
    methods
        function f = initialize_session(obj)
        % initialize_session Initialize accompanying Python session. Run at the beginning of every session.
            pyrun("import os")
            pyrun("from nowcast_lstm.LSTM import LSTM")
            pyrun("import pandas as pd")
            pyrun("import numpy as np")
            pyrun("import dill")
            pyrun("import torch")
        end

        function f = df_matlab_to_python(obj, df, python_name, date_column)
        % df_matlab_to_python Pass a MATLAB table into Python.
        % Arguments:
        % -df: MATLAB dataframe / table to pass to Python
        % -python_name: String, what name to give Python variable. Should pass the same as the MATLAB name of 'df'
        % -date_column: String, column name of the date column of the dataframe
            writetable(df, "tmp.csv");
            pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
            pyrun("os.remove('tmp.csv')");
        end

        function f = df_python_to_matlab(obj, python_df_name)
        % df_python_to_matlab Pass a Python dataframe into MATLAB.
        % Arguments:
        % -python_df_name: String, name of dataframe in python
            pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
            f = readtable('tmp.csv');
            pyrun("os.remove('tmp.csv')");
        end

        function f = gen_lstm_parameters(obj, my_map)
        % gen_lstm_parameters generate Map of default parameters for LSTM, augmented by user inputs from my_map.
        % Arguments:
        % -my_map: Map, Map containing any parameters to be changed from default

            % default values
            final_map = containers.Map;
            final_map('python_model_name') = 'model';
            final_map('data') = '';
            final_map('target_variable') = '';
            final_map('n_timesteps') = 1;
            final_map('fill_na_func') = 'np.nanmean';
            final_map('fill_ragged_edges_func') = 'np.nanmean';
            final_map('n_models') = 1;
            final_map('train_episodes') = 200;
            final_map('batch_size') = 30;
            final_map('decay') = 0.98;
            final_map('n_hidden') = 20;
            final_map('n_layers') = 2;
            final_map('dropout') = 0.0;
            final_map('criterion') = 'torch.nn.L1Loss()';
            final_map('optimizer') = 'torch.optim.Adam';
            final_map('optimizer_parameters') = "{'lr': 0.01}";

            % overwriting defaults with user provided values
            for argument = keys(my_map)
                final_map(argument{1}) = my_map(argument{1});
            end

            % reordering for proper order for LSTM function
            ordered_keys = {'python_model_name', 'data', 'target_variable', 'n_timesteps', 'fill_na_func', 'fill_ragged_edges_func', 'n_models', 'train_episodes', 'batch_size', 'decay', 'n_hidden', 'n_layers', 'dropout', 'criterion', 'optimizer', 'optimizer_parameters'};
            ordered_values = [""];
            for i = 1:length(ordered_keys)
                param_name = ordered_keys(i);
                ordered_values = [ordered_values, final_map(param_name{1})];
            end
            ordered_values = ordered_values(2:end); % dropping unneeded initial ""
            ordered_values = mat2cell(ordered_values,1,ones(1,numel(ordered_values))); % transforming to format for passing to LSTM function
            f = ordered_values;
        end

        function f = LSTM(obj, python_model_name, data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters)
        % LSTM Instantiate an LSTM model. Pass output of gen_lstm_parameters function, e.g. x = gen_lstm_parameters(my_map); LSTM(x{:})
        % Arguments:
        % -python_model_name: String, what to name the model in Python
        % -data: String, name of the training data in Python
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

            pyrun(sprintf("%s = LSTM(data=%s, target_variable='%s', n_timesteps=%s, fill_na_func=%s, fill_ragged_edges_func=%s, n_models=%s, train_episodes=%s, batch_size=%s, decay=%s, n_hidden=%s, n_layers=%s, dropout=%s, criterion=%s, optimizer=%s, optimizer_parameters=%s)", python_model_name, data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters))
        end

        function f = train(obj, python_model_name, quiet)
        % train Train an instanitated LSTM model
        % Arguments:
        % -python_model_name: String, name of model in python
        % -quiet: Boolean, whether or not to print out training epoch losses at the end of training
           if quiet
               quiet_param = "True";
           else
               quiet_param = "False";
           end
           pyrun(sprintf("%s.train(quiet=%s)", python_model_name, quiet_param))
        end

        function f = predict(obj, python_model_name, data, python_data_name, date_column, only_actuals_obs)
        % predict Get predictions from a trained LSTM model
        % Arguments:
        % -python_model_name: String, name of model in python
        % -data: MATLAB dataframe / table, the data to get predictions on
        % -python_data_name: String, what to call the data in Python, should be the same as the MATLAB name passed to `data`
        % -date_column: String, column name of the date column of the dataframe
        % -only_actuals_obs: Boolean, whether or not to predict observations without a target actual

            % duplicate of df moving functions
            function f = df_matlab_to_python(df, python_name, date_column)
               writetable(df, "tmp.csv");
               pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
               pyrun("os.remove('tmp.csv')");
            end
            function f = df_python_to_matlab(python_df_name)
               pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
               f = readtable('tmp.csv');
               pyrun("os.remove('tmp.csv')");
            end
            % duplicate of df moving functionss

           if only_actuals_obs
               only_actuals_obs_param = "True";
           else
               only_actuals_obs_param = "False";
           end
           df_matlab_to_python(data, python_data_name, date_column);
           pyrun(sprintf("tmp = %s.predict(%s, %s)", python_model_name, python_data_name, only_actuals_obs_param));
           f = df_python_to_matlab("tmp");
        end

         % save LSTM model
        function f = save_lstm(obj, python_model_name, path)
        % save_lstm Save a trained LSTM model to disk
        % Arguments:
        % -python_model_name: String, name of model in Python
        % -path: String, where to save the file. File name should end in '.pkl'
        
           pyrun(sprintf("dill.dump(%s, open('%s', mode='wb'))", python_model_name, path))
        end

        function f = load_lstm(obj, python_model_name, path)
        % load_lstm Load a trained LSTM model from disk
        % Arguments:
        % -python_model_name: String, what to name model in Python
        % -path: String, where to load the file from

           pyrun(sprintf("%s = dill.load(open('%s', 'rb', -1))", python_model_name, path))
        end

        function f = ragged_preds(obj, python_model_name, pub_lags, lag, data, python_data_name, date_column, start_date, end_date)
        % ragged_preds Get predictions on artificial vintages
        % Arguments:
        % -python_model_name: String, name of model in Python
        % -pub_lags: Array, list of periods back each input variable is set to missing. I.e. publication lag of the variable. E.g. [1 2 1]. Should be length of number of independent variables
        % -lag: Int, simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
        % -data: MATLAB dataframe / table to pass to Python
        % -python_data_name: String, what name to give Python variable. Should pass the same as the MATLAB name of 'data'
        % -date_column: String, column name of the date column of the dataframe
        % -start_date: String, optional, string in "YYYY-MM-DD" format: start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
        % -end_date: String, optional, string in "YYYY-MM-DD" format: end date of generating ragged preds.

            % duplicate of df moving functions
            function f = df_matlab_to_python(df, python_name, date_column)
               writetable(df, "tmp.csv");
               pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
               pyrun("os.remove('tmp.csv')");
            end
            function f = df_python_to_matlab(python_df_name)
               pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
               f = readtable('tmp.csv');
               pyrun("os.remove('tmp.csv')");
            end

            pub_lags = strcat("[", strjoin(string(pub_lags), ","), "]");
            df_matlab_to_python(data, python_data_name, date_column);

           % only run with start and end date if provied
            if nargin > 7
                pyrun(sprintf("tmp = %s.ragged_preds(%s, %s, %s, '%s', '%s')", python_model_name, pub_lags, string(lag), python_data_name, start_date, end_date));
            else
                pyrun(sprintf("tmp = %s.ragged_preds(%s, %s, %s)", python_model_name, pub_lags, string(lag), python_data_name));
            end
            f = df_python_to_matlab("tmp");
        end

        function f = gen_news(obj, python_model_name, target_period, old_data, new_data, date_column)
        % gen_news Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output. Make sure both the old and new dataset have the target period in them to allow for predictions and news generation.
        % Arguments:
        % -python_model_name: String, name of model in Python
        % -target_period: String, target prediction date. In form "YYYY-MM-DD"
        % -old_data: MATLAB dataframe / table, previous dataset
        % -new_data: MATLAB dataframe / table, new dataset
        % -date_column: String, column name of the date column of the dataframe

           % duplicate of df moving functions
            function f = df_matlab_to_python(df, python_name, date_column)
               writetable(df, "tmp.csv");
               pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
               pyrun("os.remove('tmp.csv')");
            end
            function f = df_python_to_matlab(python_df_name)
               pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
               f = readtable('tmp.csv');
               pyrun("os.remove('tmp.csv')");
            end

           output = containers.Map
           df_matlab_to_python(old_data, "old_df", date_column);
           df_matlab_to_python(new_data, "new_df", date_column);

           pyrun(sprintf("news = %s.gen_news('%s', %s, %s)", python_model_name, target_period, "old_df", "new_df"));
            
           pyrun("news_df = news['news']");
           pyrun("holdout_df = news['holdout_discrepency']");
           news_df = df_python_to_matlab("news_df");
            
           output('news') = news_df;
           output('old_pred') = double(pyrun("x = news['old_pred']", "x"));
           output('new_pred') = double(pyrun("x = news['new_pred']", "x"));
           output('holdout_discrepency') = double(pyrun("x = news['holdout_discrepency']", "x"));
           f = output;
        end
    end
end