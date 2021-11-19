function f = df_python_to_matlab(python_df_name)
% df_python_to_matlab Pass a Python dataframe into MATLAB.
% Arguments:
% -python_df_name: String, name of dataframe in python
    pyrun(sprintf("%s.to_csv('tmp.csv', index=False)", python_df_name));
    f = readtable('tmp.csv');
    pyrun("os.remove('tmp.csv')");
end