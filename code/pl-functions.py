def extract_folder(folderpath, add_scores=False, downsample=None):
    """
    Extract CSV data from folder and subfolders into a dataframe.

    Args:
      folderpath (str): Path to the folder containing CSV files.
      add_scores (bool, optional): Boolean to add scores.csv to the dataframe. Defaults to False.
      downsample (int, optional): Number of rows to downsample CSVs to. Defaults to None.

    Returns:
      pandas.DataFrame: DataFrame of concatenated CSV data.
    """
    import os
    import pandas as pd
    
    # Dict to store dataframes by condition  
    dfs = {'control': [], 'condition': []}

    try:
        # Handle top-level scores CSV
        if add_scores and 'scores.csv' in os.listdir(folderpath):
            scores_path = os.path.join(folderpath, 'scores.csv')  
            dfs['scores'] = pd.read_csv(scores_path)

        # Get subfolders
        subfolders = [f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, f))]

        for subfolder in subfolders:
            subfolderpath = os.path.join(folderpath, subfolder)  

            # Get list of CSV files
            files = os.listdir(subfolderpath)

            for file in files:
                filepath = os.path.join(subfolderpath, file)

                # Extract ID from filename 
                id = file.split('.')[0]

                df = pd.read_csv(filepath)

                # Downsample if needed
                if downsample:
                    df = df.sample(downsample)

                # Add ID column - this is the filename without the extension
                df['id'] = id

                # Add 'condition' column
                df['condition'] = subfolder

                # Convert 'timestamp' and 'date' to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = pd.to_datetime(df['date'])

                # Append to dict by condition
                if subfolder == 'control':
                    dfs['control'].append(df)
                else:  
                    dfs['condition'].append(df)

    except OSError:
        print(f"Error reading folder: {folderpath}")

    # concatenate dfs for each condition
    dfs['control'] = pd.concat(dfs['control'])
    dfs['condition'] = pd.concat(dfs['condition'])

    # Reset index on the final df
    df = pd.concat([dfs['control'], dfs['condition']]).reset_index(drop=True)

    # add label column
    df['label'] = 0
    df.loc[df['condition'] == 'condition', 'label'] = 1
    
    # remove old 'condition' column
    df.drop('condition', axis=1, inplace=True)

    # Final concat
    return df


def extract_full_days(df):
    """
    Extracts the records from the dataframe equal to full days,
    (where each day has exactly 1440 records).

    Parameters:
    df (pandas.DataFrame): The input dataframe containing timestamp and id columns.

    Returns:
    pandas.DataFrame: The subset of the input dataframe that contains only the records
    corresponding to full days.
    """
    # Convert timestamp to date
    df['date'] = df['timestamp'].dt.date

    # Count the number of records per day
    counts = df.groupby(['id', 'date']).size()

    # Get the dates that have 1440 records
    full_days = counts[counts == 1440].reset_index()['date']

    # Extract only the records for these dates
    df_full_days = df[df['date'].isin(full_days)]

    return df_full_days



def resample_data(df, freq, agg_func='mean'):
    """
    Resamples the given DataFrame based on the specified frequency and aggregation function.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - freq (str): The frequency at which to resample the dat. (e.g., H for hourly, D for daily, 5T for every 5 minutes),
    - agg_func (str, optional): The aggregation function to apply during resampling. Defaults to 'mean'.

    Returns:
    - df_resampled (DataFrame): The resampled DataFrame.
    """
    df_resampled = df.set_index('timestamp').groupby('id').resample(freq).agg(agg_func)
    df_resampled.reset_index(inplace=True)
    return df_resampled


def extract_full_days(df):
    """
    Extracts the records from the input DataFrame that correspond to full days, 
    where each day has exactly 1440 records.

    Parameters:
    df (DataFrame): The input DataFrame containing timestamp and id columns.

    Returns:
    DataFrame: The subset of the input DataFrame that contains only the records 
    corresponding to full days.
    """
   
    # Convert timestamp to date
    df['date'] = df['timestamp'].dt.date

    # Count the number of records per day
    counts = df.groupby(['id', 'date']).size()

    # Get the dates that have 1440 records
    full_days = counts[counts == 1440].reset_index()['date']

    # Extract only the records for these dates
    df_full_days = df[df['date'].isin(full_days)]

    return df_full_days





def normalise_data(df, columns_to_normalise):
    """
    Normalise the specified columns in the df using StandardScaler.

    Parameters:
    - df (pandas.DataFrame): The DataFrame to be normalised.
    - columns_to_normalise (list): A list of column names to be normalised.

    Returns:
    - df (pandas.DataFrame): The DataFrame with the specified columns normalised.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[columns_to_normalise] = scaler.fit_transform(df[columns_to_normalise])
    return df
