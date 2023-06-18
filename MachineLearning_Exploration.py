
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
def randomsearch(variant, metric):
    """Function that uses RepeatedStratifiedKFold to get an initial estimation for the 
    optimal hyperparameters.Adjust parameters to within the if statements to cover a wide range.
    Args:
        variant: str with the model used
        metric: str with the metric used
    """
    # set model specific parameters
    if variant == 'AdaBoost':        
        #use base_estimator to research additional parameters
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        parameters = {
            'learning_rate':[0.001,0.01,0.1,1],                     #default=1
            'n_estimators':[8,64,256],                              #default=50
            'base_estimator__max_depth':[1,2,8],                    #default=1
            'algorithm': ['SAMME', 'SAMME.R']                       #default='SAMME.R'
            }
    elif variant == 'GradientBoost':
        model = GradientBoostingClassifier()
        parameters = {
            "learning_rate": [0.001,0.01,0.1,1],                    #default=0.1
            "n_estimators":[8,64,256],                              #default=100
            "max_depth":[1,2,8]                                     #default=3
            }

    elif variant == 'RandomForest':
        model = RandomForestClassifier()
        parameters = {
            'n_estimators': [8,16,64,128,256],                      #default=100
            'max_depth': [1,2,4,8,None],                            #default=None
            'criterion': ['gini', 'entropy', 'log_loss']            #default='gini'
            }
    elif variant == 'SVM':
        model = SVC()
        parameters = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'sigmoid'],
            'gamma': [0.1, 1, 10],
            'degree': [2, 3, 4]
        }
    
    # set other search parameters
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
    n_iter=5
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
        'support': make_scorer(lambda y_true, y_pred: np.sum(y_pred == 1))
    }

    # initializing the hyperparameter search
    clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=n_iter, cv=cv, scoring=scoring, refit=False, verbose=0, random_state=1, n_jobs=-1)
    clf.fit(X_train, y_train)

    # format results in df and print hyperparameters with highest AUC
    df_results=pd.DataFrame.from_dict(clf.cv_results_)

    # Format results in df and print hyperparameters with highest AUC
    df_results = pd.DataFrame.from_dict(clf.cv_results_)
    best_index = df_results['mean_test_{}'.format(metric)].idxmax()
    best_params = df_results.loc[best_index, 'params']
    best_scores = {metric: df_results.loc[best_index, f'mean_test_{metric}'] for metric in scoring}

    # Print results
    print(f"{variant} \nBest params: {best_params} \nMetrics: {best_scores}")
    
    return df_results

# Uncumment line to run random hyperparameter search
# df_random_AB = randomsearch(variant='AdaBoost', metric='balanced_accuracy')
# df_random_GB = randomsearch(variant='GradientBoost', metric='balanced_accuracy')
# df_random_RF = randomsearch(variant='RandomForest', metric='balanced_accuracy')
df_random_SVM = randomsearch(variant='SVM', metric='balanced_accuracy')

"""Results
AdaBoost 
Best params: {'n_estimators': 256, 'learning_rate': 0.1, 'base_estimator__max_depth': 8, 'algorithm': 'SAMME.R'} 
Metrics: {'balanced_accuracy': 0.5606150793650794}

GradientBoost 
Best params: {'n_estimators': 64, 'max_depth': 1, 'learning_rate': 1} 
Metrics: {'balanced_accuracy': 0.5811507936507937}

RandomForest 
Best params: {'n_estimators': 8, 'max_depth': None, 'criterion': 'log_loss'} 
Metrics: {'balanced_accuracy': 0.5269841269841269}

SVM 
Best params: {'kernel': 'linear', 'gamma': 0.1, 'degree': 3, 'C': 10} 
Metrics: {'balanced_accuracy': 0.5540674603174603}
"""

#%% GridSearch to perform final hyperparameter tuning
def gridsearch(variant, metric):
    """Function that uses GridSearchCV to get an optimal hyperparameters.
    Adjust parameters according to the results from randomsearch().
    Args:
        variant: str with the model used
        metric: str with the metric used
    """
    # set model specific parameters    
    if variant == 'AdaBoost':        
        #use base_estimator to research additional parameters
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        parameters = {
            'learning_rate':[0.01,0.1,1],                     #default=1
            'n_estimators':[32,50,64,128],                              #default=50
            'base_estimator__max_depth':[1,4,8,16],                    #default=1
            'algorithm': ['SAMME.R']                       #default='SAMME.R'
            }
    elif variant == 'GradientBoost':
        model = GradientBoostingClassifier()
        parameters = {
            "learning_rate": [0.1,1,10],                    #default=0.1
            "n_estimators":[32,64,100,128],                              #default=100
            "max_depth":[1,2,3]                                     #default=3
            }
    elif variant == 'RandomForest':
        model = RandomForestClassifier()
        parameters = {
            'n_estimators': [4,8,16,100],                      #default=100
            'max_depth': [8,32,None],                            #default=None
            'criterion': ['gini','log_loss']            #default='gini'
            }
    elif variant == 'SVM':
        model = SVC()
        parameters = {
            'C': [5,10,20],
            'kernel': ['linear'],
            'gamma': [0.01, 0.1, 1],
            'degree': [3]
        }
        
    # set other search parameters
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

    # initializing the grid search
    clf = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, cv=cv, scoring=scoring, refit=False)
    clf = clf.fit(X_train, y_train)

    # format results in df and print hyperparameters with highest AUC
    df_results=pd.DataFrame.from_dict(clf.cv_results_)
    # print(variant,'best', refit, 'score: {score:.3f} using {params}'.format(score=clf.best_score_, params=clf.best_params_))
    
    # Format results in df and print hyperparameters with highest AUC
    df_results = pd.DataFrame.from_dict(clf.cv_results_)
    best_index = df_results['mean_test_{}'.format(metric)].idxmax()
    best_params = df_results.loc[best_index, 'params']
    best_scores = {metric: df_results.loc[best_index, f'mean_test_{metric}'] for metric in scoring}

    print(f"{variant} \nBest params: {best_params} \nMetrics: {best_scores}")
    
    return df_results

# Uncumment line to run hyperparameter search
# df_grid_AB = gridsearch(variant='AdaBoost', metric='balanced_accuracy')
# df_grid_GB = gridsearch(variant='GradientBoost', metric='balanced_accuracy')
# df_grid_RF = gridsearch(variant='RandomForest', metric='balanced_accuracy')
df_random_SVM = randomsearch(variant='SVM', metric='balanced_accuracy')

"""Results
AdaBoost 
Best params: {'algorithm': 'SAMME.R', 'base_estimator__max_depth': 1, 'learning_rate': 1, 'n_estimators': 50} 
Metrics: {'balanced_accuracy': 0.5801587301587302}

GradientBoost 
Best params: {'learning_rate': 1, 'max_depth': 1, 'n_estimators': 100} 
Metrics: {'balanced_accuracy': 0.597420634920635}

RandomForest 
Best params: {'criterion': 'gini', 'max_depth': 8, 'n_estimators': 4} 
Metrics: {'balanced_accuracy': 0.5439484126984127}

SVM 
Best params: {'kernel': 'linear', 'gamma': 0.1, 'degree': 3, 'C': 10} 
Metrics: {'balanced_accuracy': 0.5673611111111112}
"""

#%%############################# ANALYZING PERFORMANCE ##############################
def performance_metrics(variant, optimized, crossval=False):
    """ Function that calculates several performance metrics for the standard
    and optimized versions of the models (with cross-validation if crossval==True).
    variant: str with the model used
    optimized: boolean to choose which version of the model to analyze
    crossval: boolean to enable cross-validation (was not used to calculate final metrics in report)
    returns dict_scores: dict with performance metrics
    """
    #base models
    if optimized==False:
        if variant == 'AdaBoost':
            clf = AdaBoostClassifier(random_state=1)
        elif variant == 'GradientBoost':
            clf = GradientBoostingClassifier(random_state=1)
        elif variant == 'RandomForest':
            clf = RandomForestClassifier(random_state=1)
        elif variant == 'SVM':
            clf = SVC(random_state=1, probability=True)
    #optimized models
    elif optimized==True:
        if variant == 'AdaBoost':
            clf = AdaBoostClassifier(random_state=1,
                                    learning_rate=1,
                                    n_estimators=50,
                                    algorithm='SAMME.R',
                                    base_estimator=DecisionTreeClassifier(max_depth=1)
                                    )
        elif variant == 'GradientBoost':
            clf = GradientBoostingClassifier(random_state=1,
                                    learning_rate=1,
                                    max_depth=1,
                                    n_estimators=100
                                    )
        elif variant == 'RandomForest':
            clf =  RandomForestClassifier(random_state=1,
                                    criterion='gini',
                                    max_depth=8,
                                    n_estimators=4
                                    )
        elif variant == 'SVM':
            clf =  SVC(random_state=1,
                                    kernel='linear',
                                    gamma=0.1,
                                    degree=3,
                                    C=10,
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
        plt.title("ROC curve "+ variant)
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
dict_basemetrics_AB = performance_metrics(variant='AdaBoost', optimized=False)
dict_basemetrics_GB = performance_metrics(variant='GradientBoost', optimized=False)
dict_basemetrics_RF = performance_metrics(variant='RandomForest', optimized=False)
dict_basemetrics_SVM = performance_metrics(variant='SVM', optimized=False)
dict_bestmetrics_AB = performance_metrics(variant='AdaBoost', optimized=True)
dict_bestmetrics_GB = performance_metrics(variant='GradientBoost', optimized=True)
dict_bestmetrics_RF = performance_metrics(variant='RandomForest', optimized=True)
dict_bestmetrics_SVM = performance_metrics(variant='SVM', optimized=True)
df_performance_metrics = pd.DataFrame([dict_basemetrics_AB, dict_bestmetrics_AB, dict_basemetrics_GB, dict_bestmetrics_GB, dict_basemetrics_RF, dict_bestmetrics_RF, dict_basemetrics_SVM, dict_bestmetrics_SVM])
df_performance_metrics.insert(0,'model', ['AdaBoost base', 'AdaBoost best', 'GradienBoost base', 'GradientBoost best', 'RandomForest base', 'RandomForest best', 'SVM base', 'SVM best'])
df_performance_metrics

#%%########################### FEATURE IMPORTANCE ##############################
def feature_importance(variant):
    """Calculates feature importance based on Gini impurity and permutation importance"""
    if variant == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=1,
                                learning_rate=1,
                                n_estimators=50,
                                algorithm='SAMME.R',
                                base_estimator=DecisionTreeClassifier(max_depth=1)
                                )
    elif variant == 'GradientBoost':
        clf = GradientBoostingClassifier(random_state=1,
                                learning_rate=1,
                                max_depth=1,
                                n_estimators=100
                                )
    elif variant == 'RandomForest':
        clf =  RandomForestClassifier(random_state=1,
                                criterion='gini',
                                max_depth=8,
                                n_estimators=4
                                )
    elif variant == 'SVM':
        clf =  SVC(random_state=1,
                                kernel='linear',
                                gamma=0.1,
                                degree=3,
                                C=10,
                                probability=True
                                )
    if variant == 'SVM':
        clf.fit(X_train, y_train)
        #calculate ROC curve 
        prediction_proba = clf.predict_proba(X_test)
        positive_proba = prediction_proba[:, 1] #only keep instable probabilities
        
        fpr_proba,tpr_proba,_ = roc_curve(y_test,positive_proba) 


        return fpr_proba,tpr_proba
    else:
        clf.fit(X_train, y_train)
        
        #plot feature importance based on Gini impurity
        features = X_train.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)

        
        #plot feature importance based on permutation importance
        test_results = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=1, n_jobs=2)
        
        all_results = permutation_importance(
        clf, X, y, n_repeats=10, random_state=1, n_jobs=2)
        
        #test_results
        sorted_importances_idx = test_results.importances_mean.argsort()
        importances_test = pd.DataFrame(test_results.importances[sorted_importances_idx].T,
            columns=X.columns[sorted_importances_idx],)
        
        #all results
        sorted_importances_idx = all_results.importances_mean.argsort()
        importances_all = pd.DataFrame(all_results.importances[sorted_importances_idx].T,
            columns=X.columns[sorted_importances_idx],)
        
        
        #calculate ROC curve 
        prediction_proba = clf.predict_proba(X_test)
        positive_proba = prediction_proba[:, 1] #only keep instable probabilities
        
        fpr_proba,tpr_proba,_ = roc_curve(y_test,positive_proba) 

        return variant,features,importances,indices,importances_test,importances_all,fpr_proba,tpr_proba


variant_R,features_R,importances_R,indices_R,importances_test_R,importances_all_R,fpr_proba_R,tpr_proba_R = feature_importance("RandomForest")    
variant_A,features_A,importances_A,indices_A,importances_test_A,importances_all_A,fpr_proba_A,tpr_proba_A = feature_importance("AdaBoost")
variant_G,features_G,importances_G,indices_G,importances_test_G,importances_all_G,fpr_proba_G,tpr_proba_G = feature_importance("GradientBoost")
fpr_proba_SVM,tpr_proba_SVM = feature_importance("SVM")

#%%  Make ROC curves for all models 

fontsize= 14

fig = plt.figure()
fig.set_figwidth(6)
fig.set_figheight(6)
    
plt.plot(fpr_proba_R,tpr_proba_R,color="r",label="RandomForest")
plt.plot(fpr_proba_A,tpr_proba_A,color="g",label="AdaBoost")
plt.plot(fpr_proba_G,tpr_proba_G,color="b",label="GradientBoost")
plt.plot(fpr_proba_SVM,tpr_proba_SVM,color="k",label="SVM")


plt.xlabel('False Positive Rate',fontsize=fontsize)
plt.ylabel('True Positive Rate',fontsize=fontsize)
plt.title("ROC curves RandomForest, AdaBoost and GradientBoost",fontsize=fontsize)

plt.legend()
plt.show()


#%% 
def figures(variant, features, importances, indices, importances_test, importances_all, fpr_proba, tpr_proba):
    """Function that plots gini and permutation importance"""
    fontsize = 13
    top_n = 20  # Number of top features to plot

    # Sort the importances and indices based on the importances in descending order
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]

    # Plot Gini
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.title('Gini impurity based Feature Importance: ' + str(variant), fontsize=fontsize + 4)
    plt.barh(range(top_n), sorted_importances[:top_n], color='b', align='center')
    plt.yticks(range(top_n), [features[i] for i in sorted_indices][:top_n], fontsize=fontsize)

    plt.xlabel('Relative Importance', fontsize=fontsize)
    plt.tight_layout()




figures(variant_A,features_A,importances_A,indices_A,importances_test_A,importances_all_A,fpr_proba_A,tpr_proba_A)
figures(variant_G,features_G,importances_G,indices_G,importances_test_G,importances_all_G,fpr_proba_G,tpr_proba_G)
figures(variant_R,features_R,importances_R,indices_R,importances_test_R,importances_all_R,fpr_proba_R,tpr_proba_R)
# %%
