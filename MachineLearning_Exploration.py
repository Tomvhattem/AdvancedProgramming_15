
"""

Exploration of several machine learning techniques such as Adaboost, GradientBoost, Randomforest and SupportVectorMachine.
Custom implementation of the networks.

"""


#%% Importing libraries
#general libraries
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#libraries for setting up estimators
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

#libraries for hyperparameter optimization
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#libraries for performance metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, balanced_accuracy_score

from sklearn.inspection import permutation_importance

#%%
df_data = pd.read_csv("processed_data.csv")
#get label dataframe
y = df_data.iloc[:,0]
#get all features
X = df_data.loc[:, df_data.columns != 'ALDH1_inhibition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1,stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size =0.5, random_state=1,stratify=y_test) #* geen validation


############################# HYPERPARAMETER TUNING ##############################
#%% RandomSearch for initial hyperparameter tuning
def hyperparameter_randomsearch(model_variant, evaluation_metric):
    """
    Perform hyperparameter search using RepeatedStratifiedKFold to obtain an initial estimation
    of the optimal hyperparameters for a given model variant.

    Args:
        model_variant (str): The variant of the model to be used.
        evaluation_metric (str): The metric used for evaluation.

    Returns:
        pd.DataFrame: Dataframe containing the results of the hyperparameter search.
    """
    # Set model-specific parameters
    if model_variant == 'AdaBoost':
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        params = {
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'n_estimators': [8, 64, 256],
            'base_estimator__max_depth': [1, 2, 8],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    elif model_variant == 'GradientBoost':
        model = GradientBoostingClassifier()
        params = {
            "learning_rate": [0.001, 0.01, 0.1, 1],
            "n_estimators": [8, 64, 256],
            "max_depth": [1, 2, 8]
        }
    elif model_variant == 'RandomForest':
        model = RandomForestClassifier()
        params = {
            'n_estimators': [8, 16, 64, 128, 256],
            'max_depth': [1, 2, 4, 8, None],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
    elif model_variant == 'SupportVectorMachine':
        model = SVC()
        params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': [0.1, 1, 10],
            'degree': [2, 3, 4]
        }

    # Seto other parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    n_iter = 5
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'support': make_scorer(lambda y_true, y_pred: np.sum(y_pred == 1))
    }
    # Define and fit clf
    clf = RandomizedSearchCV(
        estimator=model, param_distributions=params, n_iter=n_iter,
        cv=cv, scoring=scoring, refit=False, verbose=0, random_state=1, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    df_results = pd.DataFrame.from_dict(clf.cv_results_)
    best_index = df_results['mean_test_{}'.format(evaluation_metric)].idxmax()
    best_params = df_results.loc[best_index, 'params']
    best_scores = {evaluation_metric: df_results.loc[best_index, f'mean_test_{evaluation_metric}'] for evaluation_metric in scoring}

    print(f"{model_variant} \nBest params: {best_params} \nMetrics: {best_scores}")

    return df_results

# Uncumment line to run hyperparameter randomsearch
df_random_AB = hyperparameter_randomsearch(model_variant='AdaBoost', evaluation_metric='precision')
df_random_GB = hyperparameter_randomsearch(model_variant='GradientBoost', evaluation_metric='precision')
df_random_RF = hyperparameter_randomsearch(model_variant='RandomForest', evaluation_metric='precision')
df_random_SVM = hyperparameter_randomsearch(model_variant='SupportVectorMachine', evaluation_metric='precision')

"""Results
AdaBoost 
Best params: {'n_estimators': 256, 'learning_rate': 0.01, 'base_estimator__max_depth': 2, 'algorithm': 'SAMME.R'} 
Metrics: {'balanced_accuracy': 0.7416170634920635, 'accuracy': 0.8335416666666667, 'precision': 0.8855731127743032, 'recall': 0.5118055555555555, 'f1': 0.647675691633051, 'roc_auc': 0.7416170634920635, 'support': 55.53333333333333}
GradientBoost 
Best params: {'n_estimators': 64, 'max_depth': 1, 'learning_rate': 0.1} 
Metrics: {'balanced_accuracy': 0.7492559523809523, 'accuracy': 0.8314583333333333, 'precision': 0.8384932542687249, 'recall': 0.54375, 'f1': 0.658774302741156, 'roc_auc': 0.7492559523809523, 'support': 62.333333333333336}
RandomForest 
Best params: {'n_estimators': 256, 'max_depth': 1, 'criterion': 'log_loss'} 
Metrics: {'balanced_accuracy': 0.6327876984126984, 'accuracy': 0.778125, 'precision': 0.9670411489319536, 'recall': 0.2694444444444445, 'f1': 0.4198989125208793, 'roc_auc': 0.6327876984126984, 'support': 26.733333333333334}
SupportVectorMachine 
Best params: {'kernel': 'linear', 'gamma': 0.1, 'degree': 2, 'C': 1} 
Metrics: {'balanced_accuracy': 0.7742559523809524, 'accuracy': 0.8422916666666665, 'precision': 0.8266517342460048, 'recall': 0.6041666666666667, 'f1': 0.6956099146177938, 'roc_auc': 0.7742559523809524, 'support': 70.46666666666667}
"""

#%% GridSearch to perform final hyperparameter tuning
def hyperparameter_gridsearch(model_variant, evaluation_metric):
    """
    Perform hyperparameter search using GridSearchCV to find optimal hyperparameters for a given model variant.
    Adjust the parameters based on the results from randomsearch().

    Args:
        model_variant (str): The variant of the model to be used.
        evaluation_metric (str): The metric used for evaluation.

    Returns:
        pd.DataFrame: Dataframe containing the results of the hyperparameter search.
    """
    # Set model-specific parameters
    if model_variant == 'AdaptiveBoosting':
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        parameters = {
            'learning_rate': [0.001, 0.01],
            'n_estimators': [256, 512],
            'base_estimator__max_depth': [1, 2],
            'algorithm': ['SAMME.R']
        }
    elif model_variant == 'GradientBoosting':
        model = GradientBoostingClassifier()
        parameters = {
            "learning_rate": [0.01, 0.1],
            "n_estimators": [64, 128],
            "max_depth": [1, 2]
        }
    elif model_variant == 'RandomForest':
        model = RandomForestClassifier()
        parameters = {
            'n_estimators': [256, 512],
            'max_depth': [1, 2],
            'criterion': ['log_loss']
        }
    elif model_variant == 'SupportVectorMachine':
        model = SVC()
        parameters = {
            'C': [1, 2],
            'kernel': ['linear'],
            'gamma': [0.01, 0.1],
            'degree': [2]
        }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'support': make_scorer(lambda y_true, y_pred: np.sum(y_pred == 1))
    }

    clf = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, cv=cv, scoring=scoring, refit=False)
    clf.fit(X_train, y_train)

    df_results = pd.DataFrame.from_dict(clf.cv_results_)
    best_index = df_results['mean_test_{}'.format(evaluation_metric)].idxmax()
    best_params = df_results.loc[best_index, 'params']
    best_scores = {evaluation_metric: df_results.loc[best_index, f'mean_test_{evaluation_metric}'] for evaluation_metric in scoring}

    print(f"{model_variant} \nBest params: {best_params} \nMetrics: {best_scores}")

    return df_results

df_grid_AB = hyperparameter_gridsearch(model_variant='AdaptiveBoosting', evaluation_metric='precision')
df_grid_GB = hyperparameter_gridsearch(model_variant='GradientBoosting', evaluation_metric='precision')
df_grid_RF = hyperparameter_gridsearch(model_variant='RandomForest', evaluation_metric='precision')
df_grid_RF = hyperparameter_gridsearch(model_variant='SupportVectorMachine', evaluation_metric='precision')

"""Results
AdaptiveBoosting 
Best params: {'algorithm': 'SAMME.R', 'base_estimator__max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 256} 
Metrics: {'balanced_accuracy': 0.7379960317460318, 'accuracy': 0.8329166666666666, 'precision': 0.8996585036435508, 'recall': 0.5006944444444444, 'f1': 0.641750681861637, 'roc_auc': 0.7379960317460318, 'support': 53.6}
GradientBoosting 
Best params: {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 64} 
Metrics: {'balanced_accuracy': 0.678968253968254, 'accuracy': 0.805, 'precision': 0.9640231198529956, 'recall': 0.36388888888888893, 'f1': 0.5231574316880767, 'roc_auc': 0.6789682539682541, 'support': 36.266666666666666}
RandomForest 
Best params: {'criterion': 'log_loss', 'max_depth': 1, 'n_estimators': 512} 
Metrics: {'balanced_accuracy': 0.6360615079365078, 'accuracy': 0.7802083333333333, 'precision': 0.9718772249333486, 'recall': 0.2756944444444444, 'f1': 0.4275657201904197, 'roc_auc': 0.6360615079365078, 'support': 27.266666666666666}
SupportVectorMachine 
Best params: {'C': 1, 'degree': 2, 'gamma': 0.01, 'kernel': 'linear'} 
Metrics: {'balanced_accuracy': 0.7714781746031746, 'accuracy': 0.838125, 'precision': 0.8109695644818363, 'recall': 0.6048611111111111, 'f1': 0.689881884403843, 'roc_auc': 0.7714781746031746, 'support': 71.93333333333334}
"""

#%%############################# ANALYZING PERFORMANCE ##############################
def performance_metrics(model_variant, optimized, crossval=False):
    """
    Calculate various performance metrics for the standard and optimized versions of the models.
    
    Args:
        model_variant (str): The variant of the model to be used.
        optimized (bool, optional): Choose the optimized version of the model if True. Default is False.
        cross_validation (bool, optional): Enable cross-validation if True. Default is False.
    
    Returns:
        dict: Dictionary with performance metrics.
    """
    # Base models
    if not optimized:
        if model_variant == 'AdaptiveBoosting':
            clf = AdaBoostClassifier(random_state=1)
        elif model_variant == 'GradientBoosting':
            clf = GradientBoostingClassifier(random_state=1)
        elif model_variant == 'RandomForest':
            clf = RandomForestClassifier(random_state=1)
        elif model_variant == 'SupportVectorMachine':
            clf = SVC(random_state=1, probability=True)
    # Optimized models
    else:
        if model_variant == 'AdaptiveBoosting':
            clf = AdaBoostClassifier(random_state=1,
                                     learning_rate=0.01,
                                     n_estimators=256,
                                     algorithm='SAMME.R',
                                     base_estimator=DecisionTreeClassifier(max_depth=2)
                                    )
        elif model_variant == 'GradientBoosting':
            clf = GradientBoostingClassifier(random_state=1,
                                             learning_rate=0.01,
                                             max_depth=2,
                                             n_estimators=64
                                            )
        elif model_variant == 'RandomForest':
            clf = RandomForestClassifier(random_state=1,
                                         criterion='log_loss',
                                         max_depth=1,
                                         n_estimators=512
                                        )
        elif model_variant == 'SupportVectorMachine':
            clf = SVC(random_state=1,
                      kernel='linear',
                      gamma=0.01,
                      degree=2,
                      C=1,
                      probability=True
                     )
    
    if crossval == False:
        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)
        prediction_proba = clf.predict_proba(X_test)
        positive_proba = prediction_proba[:, 1] #only keep instable probabilities
        
        #calculate ROC curve and AUC
        fpr_proba,tpr_proba,_ = roc_curve(y_test,positive_proba) 
        AUC = roc_auc_score(y_test,positive_proba)

        #plot ROC curve
        plt.plot(fpr_proba,tpr_proba)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC curve "+ model_variant)
        plt.show()

        #calculate other performance metrics
        tn, fp, fn, tp = confusion_matrix(y_test,prediction).ravel()

        performance_metrics={
            'balanced_accuracy': balanced_accuracy_score(y_test,prediction),
            'AUC':AUC,
            'tn':tn,
            'fp':fp,
            'fn':fn,
            'tp':tp,
            'accuracy':accuracy_score(y_test,prediction),
            'precision':precision_score(y_test,prediction),
            'recall':recall_score(y_test,prediction),
            'f1':f1_score(y_test,prediction)
            }
        return performance_metrics
    
    elif crossval == True:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
        scoring=['accuracy', 'precision', 'recall','f1', 'roc_auc']
        dict_scores = cross_validate(clf, X, y, cv=cv, scoring=scoring) 
        #take mean of scores
        for key in dict_scores:
            dict_scores[key]=dict_scores[key].mean()
        return dict_scores

# recieve performance scores from base and best versions of model and combine data in df_performance_metrics    
dict_basemetrics_AB = performance_metrics(model_variant='AdaptiveBoosting', optimized=False)
dict_basemetrics_GB = performance_metrics(model_variant='GradientBoosting', optimized=False)
dict_basemetrics_RF = performance_metrics(model_variant='RandomForest', optimized=False)
dict_basemetrics_SVM = performance_metrics(model_variant='SupportVectorMachine', optimized=False)
dict_bestmetrics_AB = performance_metrics(model_variant='AdaptiveBoosting', optimized=True)
dict_bestmetrics_GB = performance_metrics(model_variant='GradientBoosting', optimized=True)
dict_bestmetrics_RF = performance_metrics(model_variant='RandomForest', optimized=True)
dict_bestmetrics_SVM = performance_metrics(model_variant='SupportVectorMachine', optimized=True)
df_performance_metrics = pd.DataFrame([dict_basemetrics_AB, dict_bestmetrics_AB, dict_basemetrics_GB, dict_bestmetrics_GB, dict_basemetrics_RF, dict_bestmetrics_RF, dict_basemetrics_SVM, dict_bestmetrics_SVM])
df_performance_metrics.insert(0,'model', ['AdaptiveBoosting base', 'AdaptiveBoosting best', 'GradientBoosting base', 'GradientBoosting best', 'RandomForest base', 'RandomForest best', 'SupportVectorMachine base', 'SupportVectorMachine best'])
df_performance_metrics

#%%########################### FEATURE IMPORTANCE ##############################
def feature_importance(model_variant):
    """
    Calculate feature importance based on Gini impurity and permutation importance.
    
    Args:
        model_variant (str): The variant of the model to be used.
    
    Returns:
        Various variables containing feature importance and ROC curve data.
    """
    if model_variant == 'AdaptiveBoosting':
        clf = AdaBoostClassifier(random_state=1,
                                    learning_rate=0.01,
                                    n_estimators=256,
                                    algorithm='SAMME.R',
                                    base_estimator=DecisionTreeClassifier(max_depth=2)
                                )
    elif model_variant == 'GradientBoosting':
        clf = GradientBoostingClassifier(random_state=1,
                                            learning_rate=0.01,
                                            max_depth=2,
                                            n_estimators=64
                                        )
    elif model_variant == 'RandomForest':
        clf = RandomForestClassifier(random_state=1,
                                        criterion='log_loss',
                                        max_depth=1,
                                        n_estimators=512
                                    )
    elif model_variant == 'SupportVectorMachine':
        clf = SVC(random_state=1,
                    kernel='linear',
                    gamma=0.01,
                    degree=2,
                    C=1,
                    probability=True
                    )
    
    if model_variant == 'SupportVectorMachine':
        clf.fit(X_train, y_train)
        
        prediction_proba = clf.predict_proba(X_test)
        positive_proba = prediction_proba[:, 1]
        
        fpr_proba, tpr_proba, _ = roc_curve(y_test, positive_proba)
        
        return fpr_proba, tpr_proba
    
    else:
        clf.fit(X_train, y_train)
        
        features = X_train.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)

        test_results = permutation_importance(
            clf, X_test, y_test, n_repeats=10, random_state=1, n_jobs=2)

        all_results = permutation_importance(
            clf, X, y, n_repeats=10, random_state=1, n_jobs=2)

        sorted_importances_idx = test_results.importances_mean.argsort()
        importances_test = pd.DataFrame(test_results.importances[sorted_importances_idx].T,
                                        columns=X.columns[sorted_importances_idx])

        sorted_importances_idx = all_results.importances_mean.argsort()
        importances_all = pd.DataFrame(all_results.importances[sorted_importances_idx].T,
                                       columns=X.columns[sorted_importances_idx])

        prediction_proba = clf.predict_proba(X_test)
        positive_proba = prediction_proba[:, 1]

        fpr_proba, tpr_proba, _ = roc_curve(y_test, positive_proba)

        return model_variant, features, importances, indices, importances_test, importances_all, fpr_proba, tpr_proba


variant_R, features_R, importances_R, indices_R, importances_test_R, importances_all_R, fpr_proba_R, tpr_proba_R = feature_importance("RandomForest")
variant_A, features_A, importances_A, indices_A, importances_test_A, importances_all_A, fpr_proba_A, tpr_proba_A = feature_importance("AdaptiveBoosting")
variant_G, features_G, importances_G, indices_G, importances_test_G, importances_all_G, fpr_proba_G, tpr_proba_G = feature_importance("GradientBoosting")
fpr_proba_SVM, tpr_proba_SVM = feature_importance("SupportVectorMachine")


#%%  Make combined ROC curves for all models 

fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(6)
    
plt.plot(fpr_proba_R,tpr_proba_R,color="r",label="RandomForest")
plt.plot(fpr_proba_A,tpr_proba_A,color="g",label="AdaptiveBoosting")
plt.plot(fpr_proba_G,tpr_proba_G,color="b",label="GradientBoosting")
plt.plot(fpr_proba_SVM,tpr_proba_SVM,color="k",label="SupportVectorMachine")


plt.xlabel('False Positive Rate',fontsize=14)
plt.ylabel('True Positive Rate',fontsize=14)
plt.title("ROC curves for 4 basic classifiers",fontsize=14)

plt.legend()
plt.show()

#%% 
def figures(variant, features, importances):
    """Function that plots gini and permutation importance"""
    fontsize = 13
    top_n = 10  # Number of top features to plot

    # Sort the importances and indices based on the importances in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]

    # Plot Gini
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.title('Feature Importance: ' + str(variant), fontsize=fontsize + 4)
    plt.barh(range(top_n), sorted_importances[:top_n], color='b', align='center')
    plt.yticks(range(top_n), [features[i] for i in sorted_indices][:top_n], fontsize=fontsize)

    plt.xlabel('Relative Importance', fontsize=fontsize)
    plt.tight_layout()

figures(variant_A,features_A,importances_A)
figures(variant_G,features_G,importances_G)
figures(variant_R,features_R,importances_R)
# %%
