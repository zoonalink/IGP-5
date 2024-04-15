import pandas as pd
import calendar
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

## extract data into df

def extract_from_folder(folderpath, downsample=None, save_to_csv=False, output_csv_path=None):
    """
    Extract CSV data from folder and subfolders into a dataframe.

    Args:
      folderpath (str): folder containing CSV files.
      downsample (int, optional): number of rows to downsample CSVs to. Defaults to None.
      save_to_csv (bool, optional): save the updated df to a CSV file? defaults to False.
      output_csv_path (str, optional): csv filepath. required if save_to_csv is True.

    Returns:
      pandas.DataFrame: DataFrame of concatenated CSV data.
    """
    import os
    import pandas as pd
    
    # dict to store dataframes by condition  
    dfs = {'control': [], 'condition': []}

    try:
        # subfolders
        subfolders = [f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, f))]

        for subfolder in subfolders:
            subfolderpath = os.path.join(folderpath, subfolder)  

            # list of CSV files
            files = os.listdir(subfolderpath)

            for file in files:
                filepath = os.path.join(subfolderpath, file)

                # extract ID from filename 
                id = file.split('.')[0]

                df = pd.read_csv(filepath)

                # optional downsample 
                if downsample:
                    df = df.sample(downsample)

                # ID column - this is the filename without the extension
                df['id'] = id

                # 'condition' column
                df['condition'] = subfolder

                # convert 'timestamp' and 'date' to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = pd.to_datetime(df['date'])

                # append to dict by condition
                if subfolder == 'control':
                    dfs['control'].append(df)
                else:  
                    dfs['condition'].append(df)

    except OSError:
        print(f"Error reading folder: {folderpath}")

    # concatenate dfs for each condition
    dfs['control'] = pd.concat(dfs['control'])
    dfs['condition'] = pd.concat(dfs['condition'])

    # reset index on the final df
    df = pd.concat([dfs['control'], dfs['condition']]).reset_index(drop=True)

    # add label column
    df['label'] = 0
    df.loc[df['condition'] == 'condition', 'label'] = 1
    
    # remove old 'condition' column
    df.drop('condition', axis=1, inplace=True)


    try:
        if save_to_csv:
            if output_csv_path:
                df.to_csv(output_csv_path, index=False)
                print(f"df saved to {output_csv_path}")
            else:
                print("Error: Please provide an output CSV path.")
        
        
        return df
    except OSError:
        print("Error saving to CSV.")


## extract full days from df (i.e. where 1440 rows per id per day)

def preprocess_full_days(df, save_to_csv=False, output_csv_path=None, print_info=False):
    """
    Extracts full days from a dataframe.

    Args::
    df (DataFrame): input df.
    save_to_csv (bool, optional): save the updated df to a CSV file? defaults to False.
    output_csv_path (str, optional): csv filepath. required if save_to_csv is True.
    print_info (bool, optional): print info about the df. defaults to True.

    Returns:
    DataFrame: df containing only full days (1440 rows per day).

    """
    

    # group by id and date, count rows, and filter where count equals 1440
    full_days_df = df.groupby(['id', 'date']).filter(lambda x: len(x) == 1440)

    # set index to timestamp
    #full_days_df.set_index(['timestamp'], inplace=True)
    
    if print_info:
        # print id and date combinations that don't have 1440 rows
        not_full_days = df.groupby(['id', 'date']).size().reset_index(name='count').query('count != 1440')
        print("\nid and date combinations that don't have 1440 rows and have been removed:\n")
        print(not_full_days)

        # print info
        print("\nfull_days_df info:\n")
        print(full_days_df.info())

        #print full days per id
        print("\nfull days per id:\n")
        print(full_days_df.groupby('id').size()/1440)

        # print min number of days
        print("\nmin number of days per id:\n")
        print(full_days_df.groupby('id').size().min()/1440)
        

    try:
        if save_to_csv:
            if output_csv_path:
                full_days_df.to_csv(output_csv_path, index=False)
                print(f"df saved to {output_csv_path}")
            else:
                print("Error: Please provide an output CSV path.")
        
        
        return full_days_df
    except OSError:
        print("Error saving to CSV.")

    return full_days_df


## extract days per scores

import warnings

# ignore all warnings
#warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def extract_days_per_scores(df, scores_csv_path='..\\data\\depresjon\\scores.csv', save_to_csv=False, output_csv_path=None, min_days=None, exact_days=None):
    """
    Extract the number of days per ID from the 'scores' data.

    Args:
        df (pd.DataFrame): df containing the 'id' column.
        scores_csv_path (str, optional): path to the 'scores' CSV file. Defaults to '..\\data\\depresjon\\scores.csv'.
        save_to_csv (bool, optional): save the updated df to a CSV file? Defaults to True.
        output_csv_path (str, optional): csv filepath. Required if save_to_csv is True.
        min_days (int, optional): drop rows where 'days' column from 'scores.csv' is less than this value.
        exact_days (int, optional): keep only the specified number of days per ID.

    Returns:
        pd.DataFrame: df with the specified number of days per ID based on 'scores'.

    Raises:
        ValueError: If each ID does not have at least min_days or exact_days (if specified).
    """
    # scores from the CSV file
    scores_df = pd.read_csv(scores_csv_path)

    # merge scores with the df based on the 'id' column
    merged_df = pd.merge(df, scores_df, left_on='id', right_on='number', how='left')

    # filter rows to keep the specified minimum number of days
    if min_days is not None:
        df_filtered = merged_df.groupby('id', group_keys=False, as_index=False, sort=False).filter(lambda group: group['days'].min() >= min_days)
    else:
        df_filtered = merged_df

    # keep only the specified exact number of days per ID (if provided)
    if exact_days is not None:
        df_filtered = (
            df_filtered.sort_values(['id', 'days'])
            .groupby('id', group_keys=False, as_index=False)
            .apply(lambda group: group.iloc[:exact_days * 1440])
            .reset_index(drop=True)
        )

    # assert that each ID has at least min_days and equals exact_days (if specified)
    if min_days is not None:
        assert all(df_filtered.groupby('id')['days'].min() >= min_days), "Some IDs have fewer than the minimum number of days."
    if exact_days is not None:
        assert all(df_filtered.groupby('id')['days'].count() == exact_days * 1440), "Some IDs do not have the exact number of days."

    # drop cols number, days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2
    cols = ['number', 'days', 'gender', 'age', 'afftype', 'melanch', 'inpatient', 'edu', 'marriage', 'work', 'madrs1', 'madrs2']
    df_filtered.drop(cols, axis=1, inplace=True)

    # save to CSV if save_to_csv
    if save_to_csv:
        if output_csv_path:
            df_filtered.to_csv(output_csv_path, index=False)
            print(f"\n\ndf saved to {output_csv_path}")
        else:
            print("Error: Please provide an output CSV path.")

    return df_filtered

## add scores

def add_scores(df, scores_df, merge_on_df='id', merge_on_scores='number', save_to_csv=False, output_csv_path=None, include_all_labels=True):
    """
    Adds scores data.

    Args:
        df (pd.DataFrame): extracted df.
        scores_df (pd.DataFrame):  scores df
        merge_on_df (str, optional): col in extracted df to merge on. defaults to 'id'.
        merge_on_scores (str, optional): col in scores df to merge on. defaults to 'number'.
        save_to_csv (bool, optional): save the updated df to a CSV file? defaults to False.
        output_csv_path (str, optional): csv filepath. required if save_to_csv is True.
        include_all_labels (bool, optional): include all labels. defaults to True. set to False to filter only label=1.

    Returns:
        pd.DataFrame: updated df with scores added.
    """

    import pandas as pd
    try:
        # merge based on specified columns
        merged_df = pd.merge(df, scores_df, left_on=merge_on_df, right_on=merge_on_scores, how='left')
        # Fill missing values with NaN
        merged_df.fillna(value=pd.NA, inplace=True)

        # filter rows based on label
        if 'label' in merged_df.columns:
            if not include_all_labels:
                merged_df = merged_df[merged_df['label'] == 1]

        if save_to_csv:
            if output_csv_path:
                merged_df.to_csv(output_csv_path, index=False)
                print(f"Updated df saved to {output_csv_path}")
            else:
                print("Error: Please provide an output CSV path.")
        return merged_df
    except KeyError:
        print(f"Error: '{merge_on_df}' column not found in the specified DataFrames.")


#

def preprocess_and_calculate_features(df, group_by_id=True, random_state=5):
    """
    Preprocesses the input dataframe and calculates features for machine learning.

    Args:
        df (pandas.DataFrame): The input dataframe.
        group_by_id (bool, optional): Whether to group the dataset by 'id' before splitting. Defaults to True.
        random_state (int, optional): Random seed for reproducibility. Defaults to 5.

    Returns:
        tuple: A tuple containing the preprocessed training and testing data.
            X_train (pandas.DataFrame): The training features.
            y_train (pandas.Series): The training labels.
            X_test (pandas.DataFrame): The testing features.
            y_test (pandas.Series): The testing labels.
    """
    df = df.copy()

    # split the dataset into training and testing sets
    if group_by_id:
        # group by 'id' and split
        train_ids, test_ids = train_test_split(df['id'].unique(), test_size=0.2, random_state=random_state)
        train_df = df[df['id'].isin(train_ids)]
        test_df = df[df['id'].isin(test_ids)]
    else:
        # split without grouping by 'id'
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    

    # mean and standard deviation using only the training set
    mean_values = train_df.loc[:, 'min_00':'min_59'].mean()
    std_values = train_df.loc[:, 'min_00':'min_59'].std()

    # standardize each minute column using the mean and std from the training set
    for minute in range(60):
        column_name = f'min_{minute:02d}'

        train_df.loc[:, column_name] = (train_df[column_name] - mean_values[minute]) / std_values[minute]
        test_df.loc[:, column_name] = (test_df[column_name] - mean_values[minute]) / std_values[minute]

    #  features for both training and testing sets
    train_df = calculate_all_features(train_df)
    test_df = calculate_all_features(test_df)

    # drop unnecessary columns
    train_df = drop_columns(train_df)
    test_df = drop_columns(test_df)

    # separate features and labels
    X_train = train_df.drop(['label', 'id'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label', 'id'], axis=1)
    y_test = test_df['label']

    return X_train, y_train, X_test, y_test

def drop_columns(df):
    # Drop minute columns
    df = df.drop(columns=[f'min_{minute:02d}' for minute in range(60)], inplace=False)
    
    # Drop 'hour' and 'date' columns
    df = df.drop(columns=['hour', 'date'], inplace=False)
    
    return df

def calculate_all_features(df):

    df = df.copy()



    # time domain features
    df.loc[:, 'TDmean'] = df.iloc[:, 4:].mean(axis=1)
    df.loc[:, 'TDmedian'] = df.iloc[:, 4:].median(axis=1)
    df.loc[:, 'TDstd'] = df.iloc[:, 4:].std(axis=1)
    df.loc[:, 'TDvariance'] = df.iloc[:, 4:].var(axis=1)
    df.loc[:, 'TDmin'] = df.iloc[:, 4:].min(axis=1)
    df.loc[:, 'TDmax'] = df.iloc[:, 4:].max(axis=1)

    # trimmed mean 20%
    df.loc[:, 'TDkurtosis'] = df.iloc[:, 4:].apply(lambda row: kurtosis(row), axis=1)
    # df['TDskewness'] = df.iloc[:, 4:].apply(lambda row: skew(row), axis=1)
    df.loc[:, 'TDcoefficient_of_variance'] = df['TDstd'] / df['TDmean']
    df.loc[:, 'TDinterquartile_range'] = df.iloc[:, 4:].apply(lambda row: np.percentile(row, 75) - np.percentile(row, 25), axis=1)



    return df