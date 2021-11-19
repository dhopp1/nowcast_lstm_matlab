function f = gen_news(python_model_name, target_period, old_data, new_data)
% gen_news Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output. Make sure both the old and new dataset have the target period in them to allow for predictions and news generation.
% Arguments:
% -python_model_name: String, name of model in Python
% -target_period: String, target prediction date. In form "YYYY-MM-DD"
% -old_data: MATLAB dataframe / table, previous dataset
% -new_data: MATLAB dataframe / table, new dataset

   output = containers.Map;
   python_old_df = df_matlab_to_python(old_data);
   python_new_df = df_matlab_to_python(new_data);

   pyrun(sprintf("news = %s.gen_news('%s', %s, %s)", python_model_name, target_period, python_old_df, python_new_df));
    
   pyrun("news_df = news['news']");
   news_df = df_python_to_matlab("news_df");
    
   output('news') = news_df;
   output('old_pred') = double(pyrun("x = news['old_pred']", "x"));
   output('new_pred') = double(pyrun("x = news['new_pred']", "x"));
   output('holdout_discrepency') = double(pyrun("x = news['holdout_discrepency']", "x"));
   f = output;
end