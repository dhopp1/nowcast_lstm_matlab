function f = load_lstm(path)
% load_lstm Load a trained LSTM model from disk
% Arguments:
% -path: String, where to load the file from
    python_model_name = char(randi([33 126],1,40)); % randomly generated name for Python variable, stored in output of this function
    python_model_name = python_model_name(isstrprop(python_model_name,'alpha'));
    pyrun(sprintf("%s = dill.load(open('%s', 'rb', -1))", python_model_name, path));
    f = python_model_name;
end