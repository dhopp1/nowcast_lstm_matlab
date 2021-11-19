function f = save_lstm(python_model_name, path)
% save_lstm Save a trained LSTM model to disk
% Arguments:
% -python_model_name: String, name of model in Python
% -path: String, where to save the file. File name should end in '.pkl'

   pyrun(sprintf("dill.dump(%s, open('%s', mode='wb'))", python_model_name, path));
end