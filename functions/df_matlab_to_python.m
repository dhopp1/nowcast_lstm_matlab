function f = df_matlab_to_python(df)
% df_matlab_to_python Pass a MATLAB table into Python.
% Arguments:
% -df: MATLAB dataframe / table to pass to Python
    python_name = char(randi([33 126],1,40)); % randomly generated name for Python variable, stored in output of this function
    python_name = python_name(isstrprop(python_name,'alpha'));
    date_column = get_date_col(df);
    writetable(df, "tmp.csv");
    pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", python_name, date_column));
    pyrun("os.remove('tmp.csv')");
    f = python_name;
end