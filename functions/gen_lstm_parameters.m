function f = gen_lstm_parameters(my_map)
% gen_lstm_parameters generate Map of default parameters for LSTM, augmented by user inputs from my_map.
% Arguments:
% -my_map: Map, Map containing any parameters to be changed from default

    % default values
    final_map = containers.Map;
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

    % making sure data is in python
    final_map('data') = df_matlab_to_python(final_map('data'));

    % reordering for proper order for LSTM function
    ordered_keys = {'data', 'target_variable', 'n_timesteps', 'fill_na_func', 'fill_ragged_edges_func', 'n_models', 'train_episodes', 'batch_size', 'decay', 'n_hidden', 'n_layers', 'dropout', 'criterion', 'optimizer', 'optimizer_parameters'};
    ordered_values = [""];
    for i = 1:length(ordered_keys)
        param_name = ordered_keys(i);
        ordered_values = [ordered_values, final_map(param_name{1})];
    end
    ordered_values = ordered_values(2:end); % dropping unneeded initial ""
    ordered_values = mat2cell(ordered_values,1,ones(1,numel(ordered_values))); % transforming to format for passing to LSTM function
    f = ordered_values;
end