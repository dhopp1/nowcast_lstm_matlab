classdef nowcast_lstm_matlab
    methods
        % initialize python session
        function f = initialize_session(obj)
            pyrun("import os")
            pyrun("from nowcast_lstm.LSTM import LSTM")
            pyrun("import pandas as pd")
            pyrun("import numpy as np")
            pyrun("import dill")
            pyrun("import torch")
        end

        % move a matlab table into python
        function f = df_matlab_to_python(obj, df, python_name, date_column)
            writetable(df, "tmp.csv");
            pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
            pyrun("os.remove('tmp.csv')");
        end

         % move a python table into matlab
         function f = df_python_to_matlab(obj, python_df_name)
            pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
            f = readtable('tmp.csv');
            pyrun("os.remove('tmp.csv')");
         end

         % generate arguments for LSTM to get around no named parameter
         % defaults
         function f = gen_lstm_parameters(obj, my_map)
            % default values
            final_map = containers.Map;
            final_map('python_model_name') = 'model';
            final_map('data') = '';
            final_map('target_variable') = '';
            final_map('n_timesteps') = '';
            final_map('fill_na_func') = 'np.nanmean';
            final_map('fill_ragged_edges_func') = 'np.nanmean';
            final_map('n_models') = '1';
            final_map('train_episodes') = '200';
            final_map('batch_size') = '30';
            final_map('decay') = '0.98';
            final_map('n_hidden') = '20';
            final_map('n_layers') = '2';
            final_map('dropout') = '0.0';
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

         % LSTM model
         function f = LSTM(obj, python_model_name, data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters)
            pyrun(sprintf("%s = LSTM(data=%s, target_variable='%s', n_timesteps=%s, fill_na_func=%s, fill_ragged_edges_func=%s, n_models=%s, train_episodes=%s, batch_size=%s, decay=%s, n_hidden=%s, n_layers=%s, dropout=%s, criterion=%s, optimizer=%s, optimizer_parameters=%s)", python_model_name, data, target_variable, n_timesteps, fill_na_func, fill_ragged_edges_func, n_models, train_episodes, batch_size, decay, n_hidden, n_layers, dropout, criterion, optimizer, optimizer_parameters))
         end

         % train LSTM model
         function f = train(obj, python_model_name, quiet)
            if quiet
                quiet_param = "True";
            else
                quiet_param = "False";
            end
            pyrun(sprintf("%s.train(quiet=%s)", python_model_name, quiet_param))
         end

         % get predictions from trained model
         function f = predict(obj, python_model_name, data, python_data_name, date_column, only_actuals_obs)
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
            pyrun(sprintf("dill.dump(%s, open('%s', mode='wb'))", python_model_name, path))
         end

         % load LSTM model
         function f = load_lstm(obj, python_model_name, path)
            pyrun(sprintf("%s = dill.load(open('%s', 'rb', -1))", python_model_name, path))
         end

         % ragged predictions
         function f = ragged_preds(obj, python_model_name, pub_lags, lag, data, python_data_name, date_column, start_date, end_date)
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

         % generate news
         function f = gen_news(obj, python_model_name, target_period, old_data, old_data_python_name, new_data, new_data_python_name, date_column)
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
            df_matlab_to_python(old_data, old_data_python_name, date_column);
            df_matlab_to_python(new_data, new_data_python_name, date_column);

            pyrun(sprintf("news = %s.gen_news('%s', %s, %s)", python_model_name, target_period, old_data_python_name, new_data_python_name));
            
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