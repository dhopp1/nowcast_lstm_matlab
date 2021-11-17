% LSTM library requirements
%pe = pyenv('Version', path_to_python);
addpath('~/dhopp1/nowcast_lstm_matlab/');
LSTM = nowcast_lstm_matlab;

% data read
data = readtable("~/unctad/nowcast_data_update/output/2021-10-19_database_tf.csv");
data = data(:, ["date" "x_world" "x_nl" "x_de" "x_cn"]);
data = make_double(data);

% LSTM library functionality
LSTM.initialize_session();
LSTM.df_matlab_to_python(data, "data", "date");

new = LSTM.df_python_to_matlab("data");
%head(new)

my_params = containers.Map;
my_params('python_model_name') = 'model';
my_params('data') = 'data';
my_params('target_variable') = 'x_world';
my_params('n_timesteps') = '12';
my_params('n_models') = '1';
my_params('train_episodes') = '50';

params = LSTM.gen_lstm_parameters(my_params);

LSTM.LSTM(params{:});
LSTM.train("model", true);

preds = LSTM.predict("model", data, "data", "date", true);

LSTM.save_lstm("model", "test.pkl");
LSTM.load_lstm("new_model", "test.pkl");

preds2 = LSTM.predict("new_model", data, "data", "date", true);

rag_preds = LSTM.ragged_preds("model", [1 2 3], -3, data, "data", "date");
rag_preds2 = LSTM.ragged_preds("model", [1 2 3], -3, data, "data", "date", "2019-01-01", "2020-01-01");


old_data = data
old_data(end-4:end, 3) = {nan}
old_data(end-4:end, 4) = {nan}
news = LSTM.gen_news("model", "2021-06-01", old_data, "old_data", data, "data", "date");

function df = make_double(df)
    for k = 2:size(df, 2)
        col_name = df.Properties.VariableNames{k};
        if not(isnumeric(df.(col_name)))
            df.(col_name) = str2double(df.(col_name));
        end
    end
end