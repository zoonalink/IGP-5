## the functions below are used in`wiki-08-base.ipynb` which is a recreation of Garcia et al.'s Depresjon baseline models

import pandas as pd
import calendar
from matplotlib import pyplot as plt
import seaborn as sns

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

def extract_days_per_scores(df, scores_csv_path='..\data\depresjon\scores.csv', save_to_csv=False, output_csv_path=None):
    """
    Extract the number of days per ID from the 'scores' data.

    Args:
        df (pd.DataFrame): df containing the 'id' column.
        scores_csv_path (str, optional): path to the 'scores' CSV file. Defaults to '..\data\depresjon\scores.csv'.
        save_to_csv (bool, optional): save the updated df to a CSV file? Defaults to True.
        output_csv_path (str, optional): csv filepath. Required if save_to_csv is True.
        

    Returns:
        pd.DataFrame: df with the specified number of days per ID based on 'scores'.
    """
    # scores from the CSV file
    scores_df = pd.read_csv(scores_csv_path)

    # merge scores with the df based on the 'id' column
    merged_df = pd.merge(df, scores_df, left_on='id', right_on='number', how='left')

    # filter rows to keep the specified number of days
    df_filtered = merged_df.groupby('id', group_keys=False, as_index=False, sort=False).apply(lambda group: group.head(group['days'].min() * 1440)).reset_index(drop=True)

    # drop cols number, days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2
    cols = ['number', 'number', 'days', 'gender', 'age', 'afftype', 'melanch', 'inpatient', 'edu', 'marriage', 'work', 'madrs1', 'madrs2']
    df_filtered.drop(cols, axis=1, inplace=True)
    

    # save to CSV
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
        return merged_df
    

## heatmap - plotly singles
    
def plotly_heatmap(df, title="Heatmap of normalised activity by day of week and hour of day", palette='Reds256', normalised=True):
       """
       Generate a heatmap using Plotly based on the given DataFrame.

       Parameters:
       - df (pandas.DataFrame): The DataFrame containing the data to be plotted.
       - title (str): The title of the heatmap (default: "Heatmap of normalised activity by day of week and hour of day").
       - palette (str): The color palette to be used for the heatmap (default: 'Reds256').
       - normalised (bool): Whether to use the normalised activity values or the raw activity values (default: True).

       Returns:
       None
       """
       from bokeh.plotting import figure, show
       from bokeh.io import output_notebook
       from bokeh.models import LinearColorMapper, BasicTicker, PrintfTickFormatter, ColorBar
       from bokeh.palettes import Reds256, Blues256, viridis

       # color palette
       if palette == 'Blues256':
              colors = Blues256
       elif palette == 'viridis':
              colors = viridis(256)
       else:
              colors = Reds256

       # reset index to create 'hour_of_day' and 'day_of_week' columns
       df_reset = df.reset_index()

       # column to plot
       column_to_plot = 'activity_norm' if normalised else 'activity'

       # color mapper
       mapper = LinearColorMapper(palette=colors, low=df_reset[column_to_plot].max(), high=df_reset[column_to_plot].min())

       # figure
       p = figure(title=title,
                        x_range=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                        y_range=[str(y) for y in range(24)],
                        x_axis_location="above", width=900, height=400,
                        tools="hover", toolbar_location='below', tooltips=[(column_to_plot, '@'+column_to_plot)])

       # rectangle glyph
       p.rect(x="day_of_week", y="hour_of_day", width=1, height=1, source=df_reset,
                 fill_color={'field': column_to_plot, 'transform': mapper},
                 line_color=None)

       # color bar
       color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                                           ticker=BasicTicker(desired_num_ticks=len(colors)),
                                           formatter=PrintfTickFormatter(format="%f"),
                                           label_standoff=6, border_line_color=None, location=(0, 0))
       p.add_layout(color_bar, 'right')

       # Show the plot
       output_notebook()
       show(p)



## heatmap pairs


def plot_heatmap_pair(df1, df2, df1_title="df1",df2_title="df2", title='ADD TITLE Heatmaps'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # heatmap 1
    sns.heatmap(df1, cmap='Reds', annot=False, fmt='.2f', cbar=False, ax=axs[0])
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Hour of the Day')
    axs[0].set_title(df1_title)
    axs[0].set_yticks(range(0, 24))
    axs[0].set_yticklabels(range(0, 24), rotation=0)  
    axs[0].set_xticks(range(7))
    axs[0].set_xticklabels(calendar.day_name, rotation=45)

    # heatmap 2
    sns.heatmap(df2, cmap='Reds', annot=False, fmt='.2f', cbar=False, ax=axs[1])
    axs[1].set_xlabel('')
    axs[1].set_ylabel('') 
    axs[1].set_title(df2_title)
    axs[1].set_yticks(range(0, 24))
    axs[1].set_xticks(range(7))
    axs[1].set_xticklabels(calendar.day_name, rotation=45)

    # add shared title
    plt.suptitle(title)

    plt.tight_layout()
    plt.show()


## plotly lines animation
    
def plotly_24_hours(df, column, values, label=None):
    """
    Plot activity norm over hour of day using Plotly.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        column (str): The column name to use for colouring the lines.
        values (list): A list of two values to use for colouring the lines.
        label (str, optional): The column name to use for filtering the data. Defaults to None.

    Returns:
        None
    """
    import plotly.graph_objects as go       
    # figure
    fig = go.Figure()

    # all traces with 0 opacity
    for id_val, id_data in df.groupby('id'):
        if label is not None and id_data[label].iloc[0] not in values:
            continue
        fig.add_trace(go.Scatter(x=id_data['hour_of_day'], y=id_data['activity_norm'],
                                 mode='lines',
                                 line=dict(color='rgba(0,0,255,0)' if id_data[column].iloc[0] == values[0] else 'rgba(255,0,0,0.1)'),
                                 showlegend=False))

    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode='lines',
                             line=dict(color='blue', width=1),
                             showlegend=True,
                             name=f'<span style="color: blue">{values[0]}</span>',
                             legendgroup=values[0],
                             visible='legendonly'))

    fig.add_trace(go.Scatter(x=[None], y=[None],
                             mode='lines',
                             line=dict(color='red', width=1),
                             showlegend=True,
                             name=f'<span style="color: red">{values[1]}</span>',
                             legendgroup=values[1],
                             visible='legendonly'))

    # empty list to store frames
    frames = []

    # counter for traces
    trace_counter = 0

    # iterate over unique IDs
    for id_val, id_data in df.groupby('id'):
        if label is not None and id_data[label].iloc[0] not in values:
            continue
        # trace for the current ID
        trace = go.Scatter(x=id_data['hour_of_day'], y=id_data['activity_norm'],
                           mode='lines',
                           line=dict(color='red' if id_data[column].iloc[0] == values[1] else 'blue', width=1))

        # append trace and update the visibility within the loop
        frames.append(go.Frame(data=[trace], traces=[trace_counter], layout=go.Layout(showlegend=True)))
        trace_counter += 1

    # frame for stopping with all lines plotted with reduced opacity
    stop_frames = [go.Frame(data=[trace.update(line=dict(color='rgba(0,0,255,0.1)')) for trace in fig.data], layout=go.Layout(showlegend=False))]

    # initial frame (all lines plotted with opacity 0.1)
    initial_frames = [go.Frame(data=[trace.update(line=dict(color='rgba(0,0,255,0.1)')) for trace in fig.data], layout=go.Layout(showlegend=False))]

    # update the layout
    layout = go.Layout(
        title="Activity Norm Over Hour of Day",
        xaxis_title="Hour of Day",
        yaxis_title="Activity Norm",
        yaxis=dict(range=[df['activity_norm'].min(), df['activity_norm'].max()]),
        updatemenus=[
            dict(
                type="buttons",
                direction="right", # buttons on right
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 300, "redraw": False},
                                      "fromcurrent": True, "transition": {"duration": 300}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]),  # pause
                    dict(label="Reset",  # reset to initial frames
                         method="animate",
                         args=[initial_frames, {"frame": {"duration": 0, "redraw": False},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}])  
                ],
                pad={"r": 10, "t": 10},  # padding around buttons
                showactive=True,
                x=0.7,  #  horizontal position of buttons
                xanchor='left',  # anchor point for x pos
                y=1.3,  # vertical position of buttons
                yanchor='top'  # anchor point for  y pos
            )
        ],
        showlegend=True
    )

    # update figure with frames and layout
    fig.update(frames=frames + initial_frames, layout=layout)

    # display 
    fig.show()


## train / test split
    
def split_data(df, test_size=0.2, deleteXy=True, random_state=5):
    """
    Split the data into train and test sets.

    Args:
    - df: input df containing the data.
    - test_size: (float, optional, default 0.2) proportion of the data to include in the test set.
    - deleteXy: (bool, optional) If True, delete X and y from memory after splitting.

    Returns:
    - X_train: ndarray of training features.
    - y_train: ndarray of training labels.
    - X_test: ndarray of test features.
    - y_test: ndarray of test labels.
    """

    from sklearn.model_selection import train_test_split
    # separate features and labels
    X = df[['mean_activity', 'std_activity', 'pct_no_activity']].values
    y = df['label'].values

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # delete X and y from memory if deleteXy is True
    if deleteXy:
        del X, y

    return X_train, y_train, X_test, y_test

from sklearn.model_selection import train_test_split




## split dataframe for person prediction (verify and keep)

def split_data_by_person(df, test_size=0.2, random_state=5):
  """
  Split the data into train and test sets by person.

  Args:
      df: input DataFrame containing the data.
      test_size: (float, optional, default 0.2) proportion of the data to include in the test set.
      random_state: (int, optional, default 42) random state for reproducibility.

  Returns:
      X_train: ndarray of training features.
      y_train: ndarray of training labels.
      X_test: ndarray of training features.
      y_test: ndarray of training labels.
      person_ids_train: ndarray of training person IDs
      person_ids_test: ndarray of testing person IDs
  """

  # unique person IDs
  person_ids = df['id'].unique()

  # split person IDs into train and test
  train_ids, test_ids = train_test_split(person_ids, test_size=test_size, random_state=random_state)

  # assert separate person IDs
  assert len(set(train_ids) & set(test_ids)) == 0, "Person IDs found in both train and test sets"

  # train and test 
  train_df = df[df['id'].isin(train_ids)]
  test_df = df[df['id'].isin(test_ids)]

  # features and labels
  X_train = train_df[['mean_activity', 'std_activity', 'pct_no_activity']].values
  y_train = train_df['label'].values
  X_test = test_df[['mean_activity', 'std_activity', 'pct_no_activity']].values
  y_test = test_df['label'].values
  
  # person IDs in return values
  person_ids_train = train_df['id'].values
  person_ids_test = test_df['id'].values

  return X_train, y_train, X_test, y_test, person_ids_train, person_ids_test  # add person IDs




## libraries for models and metrics

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier

# cross validation
from sklearn.model_selection import cross_val_score

# metrics
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report, precision_recall_fscore_support


## zero rule classifier

from collections import Counter

class ZeroRClassifier:
    """
    ZeroRClassifier is a simple classifier that always predicts the majority class in the training data.
    """
    def fit(self, X, y):
        """
        Fit the ZeroRClassifier to the training data.

        Parameters:
        - X: The input features of shape (n_samples, n_features).
        - y: The target labels of shape (n_samples,).

        Returns:
        - None
        """
        # find the majority class
        self.majority_class = Counter(y).most_common(1)[0][0]
        
    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        - X: The input features of shape (n_samples, n_features).

        Returns:
        - y_pred: The predicted class labels of shape (n_samples,).
        """
        # return the majority class for all samples
        return [self.majority_class] * len(X)
    



## train models - single run

def single_run(X_train, X_test, y_train, y_test, models, random_seed=5):
    """
    Evaluate multiple models using various metrics and return the results.

    Args:
    - X_train (array-like): Training data features.
    - X_test (array-like): Testing data features.
    - y_train (array-like): Training data labels.
    - y_test (array-like): Testing data labels.
    - models (dict): Dictionary of model names and corresponding model objects.
    - random_seed (int): Random seed for reproducibility.

    Returns:
    - results (dict): Dictionary containing evaluation results for each model.
        The keys are the model names and the values are dictionaries containing
        the following metrics:
        - elapsed_time: Time taken to fit the model.
        - accuracy: Accuracy score.
        - precision: Precision score.
        - recall: Recall score.
        - f1: F1 score.
        - specificity: Specificity score.
        - mcc: Matthews correlation coefficient.
        - cm: Confusion matrix.
        - cr: Classification report.
        - roc_auc: ROC AUC score.
    """
    # results dictionary
    results = {}

    
    # set random seed for reproducibility
    import numpy as np
    np.random.seed(random_seed)
    
    # fit and time models
    for model_name, model in models:
        start_time = time.time()
        model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        
        y_pred = model.predict(X_test)

        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1)
        
        
        # calculate metrics
        specificity = recall[0] # recall of the negative class
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)        
        mcc = matthews_corrcoef(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        # calculate roc auc
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except (AttributeError, IndexError):
            y_pred_proba = y_pred
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # save results
        results[model_name] = {
            'elapsed_time': elapsed_time,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'mcc': mcc,
            'cm': cm,
            'cr': cr,
            'roc_auc': roc_auc,
            'y_pred': y_pred,  # add y_pred to the results dictionary
            'y_test': y_test  # add y_test to the results dictionary
        }
        
    return results


## print model results by dataset, model

def print_model_metric(all_results, dataset='all', model='all', metric='all'):
    """
    Prints the specified metric for a given dataset and model.

    Args:
    - dataset (str): The dataset name. Default is 'all'.
    - model (str): The model name. Default is 'all'.
    - metric (str): The metric name. Default is 'all'.

    Returns:
    None
    """
    metrics = ['elapsed_time', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'mcc', 'roc_auc']
    
    if dataset == 'all' and model == 'all' and metric == 'all':
        for ds in all_results.keys():
            for mdl in all_results[ds].keys():
                for mtr in metrics:
                    if mtr in ['y_pred', 'y_test']:
                        continue  # Skip printing y_pred and y_test
                    print(f"{ds} dataset {mtr} for {mdl}: {all_results[ds][mdl][mtr]}")
    elif model == 'all' and metric == 'all':
        for mdl in all_results[dataset].keys():
            for mtr in metrics:
                if mtr in ['y_pred', 'y_test']:
                    continue  # Skip printing y_pred and y_test
                print(f"{dataset} dataset {mtr} for {mdl}: {all_results[dataset][mdl][mtr]}")
    elif dataset == 'all' and metric == 'all':
        for ds in all_results.keys():
            if model in all_results[ds].keys():
                for mtr in metrics:
                    if mtr in ['y_pred', 'y_test']:
                        continue  # Skip printing y_pred and y_test
                    print(f"{ds} dataset {mtr} for {model}: {all_results[ds][model][mtr]}")
    elif metric == 'all':
        for mtr in metrics:
            if mtr in ['y_pred', 'y_test']:
                continue  # Skip printing y_pred and y_test
            print(f"{dataset} dataset {mtr} for {model}: {all_results[dataset][model][mtr]}")
    else:
        print(f"{dataset} dataset {metric} for {model}: {all_results[dataset][model][metric]}")


## print best models

def print_top_models(dataset_name, dataset_results, metric):
    """
    Prints the top 3 models for a given dataset based on a specified metric.

    Args:
        dataset_name (str): name of dataset
        dataset_results (dict): dictionary with results of different models
        metric (str): metric based of ranking

    Returns:
        None
    """
    print(f"\nTop 3 models for {dataset_name} dataset based on {metric}:")
    sorted_results = sorted(dataset_results.items(), key=lambda x: x[1][metric], reverse=True)
    for i in range(3):
        model_name, model_results = sorted_results[i]
        print(f"{i+1}. {model_name}: {model_results[metric]}")


## print fastest model

def print_second_fastest_model(dataset_name, dataset_results):
    """
    Prints the second fastest model (based on elapsed time) - so that zeroR is not considered

    Args:
        dataset_name (str): name of the dataset
        dataset_results (dict): dictionary with results.

    Returns:
        None
    """
    print(f"\nFastest model (elapsed time) for {dataset_name} dataset:")
    elapsed_time_sorted = sorted(dataset_results.items(), key=lambda x: x[1]['elapsed_time'])
    print(f"{elapsed_time_sorted[1][0]}: {elapsed_time_sorted[1][1]['elapsed_time']}")

def print_top3_fastest_model(dataset_name, dataset_results):
    """
    Prints the top 3 fastest models (based on elapsed time) - so that zeroR is not considered

    Args:
        dataset_name (str): name of the dataset
        dataset_results (dict): dictionary with results.

    Returns:
        None
    """
    print(f"\nTop 3 fastest models (elapsed time) for {dataset_name} dataset:")
    elapsed_time_sorted = sorted(dataset_results.items(), key=lambda x: x[1]['elapsed_time'])
    for i in range(1, 4):  # start from 1 to exclude zeroR
        print(f"{i}. {elapsed_time_sorted[i][0]}: {elapsed_time_sorted[i][1]['elapsed_time']}")


## person level prediction functions


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    
    Args:
    - cm: numpy.ndarray - confusion matrix to be plotted.
    - classes: list of class labels.
    - title: str, optional of the plot. Default is 'Confusion matrix'.
    - cmap: matplotlib colormap, optional - colormap to be used for the plot. Default is plt.cm.Blues.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' # integer format
    thresh = cm.max() / 2. 

    # add text annotations
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def mode_agg(x):
    """
    Calculate the mode for each group in the input data.
    Used in fit_and_evaluate_models function.

    Args:
    x (pandas.Series): The input data for which the mode needs to be calculated.

    Returns:
    The mode value for each group in the input data.
    """
    return x.mode().iloc[0]  # mode value


import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

def fit_and_person_conf_matrix(df, models, metric='accuracy', aggregation_method='mode', cv=5, random_seed=5):
    """
    Fits and evaluates multiple models using cross-validation and returns the best performing model and its confusion matrix.
    - Requires the mode_agg function.
    - Requires the plot_confusion_matrix function.

    Args:
    - df (DataFrame): The input DataFrame containing the dataset.
    - models (dict): A dictionary of model names as keys and model objects as values.
    - metric (str, optional): The evaluation metric to use. Default is 'accuracy'.
    - aggregation_method (str, optional): The method to aggregate predictions and labels at the person level. Default is 'mode'.

    Returns:
    - best_model_name (str): The name of the best performing model.
    - cm (array): The confusion matrix of the best performing model.
    """

    # set random seed for reproducibility
    import numpy as np
    np.random.seed(random_seed)

    # drop unnecessary columns
    df = df.drop(columns=['date', 'gender'])

    # separate features and target variable
    X = df.drop(columns=['id', 'label'])
    y = df['label']

    # initialise variables to store metrics
    model_metrics = {}

    # fit models and store their performances
    for model_name, model in models.items():
        # cross-validation and predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)

        # chosen metric
        if metric == 'accuracy':
            score = accuracy_score(y, y_pred)
        elif metric == 'precision':
            score = precision_score(y, y_pred)
        elif metric == 'recall':
            score = recall_score(y, y_pred)
        elif metric == 'f1':
            score = f1_score(y, y_pred)
        elif metric == 'mcc':
            score = matthews_corrcoef(y, y_pred)
        else:
            raise ValueError("Invalid metric option. Please choose from: accuracy, precision, recall, f1, mcc")

        # store metric for the model
        model_metrics[model_name] = score

    # best performing model based on chosen metric
    best_model_name = max(model_metrics, key=model_metrics.get)

    # best performing model
    best_model = models[best_model_name]

    # best model on the entire dataset
    best_model.fit(X, y)

    # predictions on the entire dataset
    y_pred_all = cross_val_predict(best_model, X, y, cv=cv)

    # aggregate predictions to person level
    df['y_pred'] = y_pred_all
    aggregated_df = df.groupby('id')['y_pred'].agg(mode_agg if aggregation_method == 'mode' else aggregation_method).reset_index()

    # aggregate labels to person level
    aggregated_labels = df.groupby('id')['label'].agg(mode_agg if aggregation_method == 'mode' else aggregation_method).reset_index()


    # both predictions and labels have the same number of samples
    if len(aggregated_df) != len(aggregated_labels):
        raise ValueError("Number of samples in aggregated predictions and labels are inconsistent")

    # confusion matrix for the best performing model
    cm = confusion_matrix(aggregated_labels['label'], aggregated_df['y_pred'])

    # confusion matrix
    plot_confusion_matrix(cm, classes=['Negative', 'Positive'], title=f'Confusion Matrix for ' + best_model_name)

    return best_model_name, cm



