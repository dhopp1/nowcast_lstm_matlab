data = readtable("~/unctad/nowcast_data_update/output/2021-10-19_database_tf.csv");
data = data(:, ["date" "x_world" "x_nl" "x_de" "x_cn"]);
data = make_double(data);

pyrun("import pandas as pd")
pyrun("import os")

df_matlab_to_python(data, "data", "date")

pyrun("print(data.head())")

% move a matlab table into python
function f = df_matlab_to_python(df, matlab_name, date_column)
    writetable(df, "tmp.csv");
    pyrun(sprintf("%s = pd.read_csv('tmp.csv', parse_dates=['%s'])", matlab_name, date_column));
    pyrun("os.remove('tmp.csv')");
end

function df = make_double(df)
    for k = 2:size(df, 2)
        col_name = df.Properties.VariableNames{k}
        if not(isnumeric(df.(col_name)))
            df.(col_name) = str2double(df.(col_name))
        end
    end
end