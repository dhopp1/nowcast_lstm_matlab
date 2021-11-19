function f = ragged_preds(python_model_name, pub_lags, lag, data, start_date, end_date)
% ragged_preds Get predictions on artificial vintages
% Arguments:
% -python_model_name: String, name of model in Python
% -pub_lags: Array, list of periods back each input variable is set to missing. I.e. publication lag of the variable. E.g. [1 2 1]. Should be length of number of independent variables
% -lag: Int, simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
% -data: MATLAB dataframe / table to pass to Python
% -start_date: String, optional, string in "YYYY-MM-DD" format: start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
% -end_date: String, optional, string in "YYYY-MM-DD" format: end date of generating ragged preds.

    pub_lags = strcat("[", strjoin(string(pub_lags), ","), "]");
    python_data_name = df_matlab_to_python(data);

   % only run with start and end date if provied
    if nargin > 4
        pyrun(sprintf("tmp = %s.ragged_preds(%s, %s, %s, '%s', '%s')", python_model_name, pub_lags, string(lag), python_data_name, start_date, end_date));
    else
        pyrun(sprintf("tmp = %s.ragged_preds(%s, %s, %s)", python_model_name, pub_lags, string(lag), python_data_name));
    end
    f = df_python_to_matlab("tmp");
end