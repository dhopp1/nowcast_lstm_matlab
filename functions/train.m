function f = train(python_model_name, quiet)
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