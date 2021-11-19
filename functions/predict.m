function f = predict(python_model_name, data, only_actuals_obs)
% predict Get predictions from a trained LSTM model
% Arguments:
% -python_model_name: String, name of model in python
% -data: MATLAB dataframe / table, the data to get predictions on
% -only_actuals_obs: Boolean, whether or not to predict observations without a target actual

   if only_actuals_obs
       only_actuals_obs_param = "True";
   else
       only_actuals_obs_param = "False";
   end
   python_data_name = df_matlab_to_python(data);
   pyrun(sprintf("tmp = %s.predict(%s, %s)", python_model_name, python_data_name, only_actuals_obs_param));
   f = df_python_to_matlab("tmp");
end