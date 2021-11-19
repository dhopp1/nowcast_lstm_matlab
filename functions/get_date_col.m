function date_col = get_date_col(df)
    for col_name = df.Properties.VariableNames
        if class(df{:,col_name}) == "datetime"
            date_col = col_name;
        end
    end
    date_col = date_col{1};
end