def process_data(data):
    """
    Delete columns with too much null values (>80%)
    For the columns whose null values take less than 50%, I fill them with the mode.
    The stem-surface, which null rate is 63.55%, is temporarily removed.
    :return:
    """
    threshold = 0.8
    high_missing_cols = data.columns[data.isnull().mean() > threshold]
    df_processed = data.drop(columns=high_missing_cols)
    medium_missing_cols = ['cap-surface', 'gill-attachment', 'gill-spacing']
    for col in medium_missing_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    df_processed = df_processed.drop('stem-surface', axis=1)
    return df_processed


