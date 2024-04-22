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
from sklearn.model_selection import cross_val_score, train_test_split

# metrics
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

## zero rule classifier

from collections import Counter

## split dataframe 

def split_data_active(df, test_size=0.2, deleteXy=True, random_state=5):
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
    X = df[['mean', 'std', '%zero', 'activeNight', 'inactiveDay', 'activeDark', 'inactiveLight']].values
    y = df['label'].values

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # delete X and y from memory if deleteXy is True
    if deleteXy:
        del X, y

    return X_train, y_train, X_test, y_test


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

# fit and evaluate models
def fit_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Fits and evaluates multiple machine learning models on the given training and testing data.

    Parameters:
    - X_train (array-like): The training input samples.
    - y_train (array-like): The target values for the training input samples.
    - X_test (array-like): The testing input samples.
    - y_test (array-like): The target values for the testing input samples.

    Returns:
    - accuracy (float): The accuracy score of the random forest model on the testing data.
    - f1 (float): The F1 score of the random forest model on the testing data.
    - conf_matrix (array-like): The confusion matrix of the random forest model on the testing data.
    - recall (float): The recall score of the random forest model on the testing data.
    - mcc (float): The Matthews correlation coefficient of the random forest model on the testing data.
    - precision (float): The precision score of the random forest model on the testing data.
    - roc_auc (float): The ROC AUC score of the random forest model on the testing data.
    - specificity (float): The specificity score of the random forest model on the testing data.
    - support (int): The support of the positive class in the testing data.
    - feature_importance_dict (dict): A dictionary containing various feature importance metrics.

    """
    # Random Forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Random Forest feature importances
    rf_feature_importances = rf_model.feature_importances_

    # Decision Tree feature importances
    dt_feature_importances = dt_model.feature_importances_

    # Permutation Importance
    permutation_result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=5)
    permutation_importances = permutation_result.importances_mean

    # Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    # evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    specificity = report['0']['recall']
    support = report['1']['support']

    # dictionary with all feature importance metrics
    feature_importance_dict = {
        "RandomForest": dict(zip(X_train.columns, rf_feature_importances)),
        "DecisionTree": dict(zip(X_train.columns, dt_feature_importances)),
        "PermutationImportance": dict(zip(X_train.columns, permutation_importances)),
        "VIF": vif_data.set_index("feature")["VIF"].to_dict()
    }

    return accuracy, f1, conf_matrix, recall, mcc, precision, roc_auc, specificity, support, feature_importance_dict