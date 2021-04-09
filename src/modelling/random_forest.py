
import matplotlib.pyplot as plt
from scipy.stats import uniform
import numpy as np
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
from scipy.stats import uniform

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics, model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def calculate_results_rf(dataframe, rf, random_grid, skfold):
    x1 = dataframe.drop('target', axis=1).values
    y1 = dataframe['target'].values

    # Busca randomizada pelos hiperparametros da regressão logistica
    #clf = RandomizedSearchCV(logistic, distributions, random_state=0)
    clf = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    search = clf.fit(x1, y1)

    search.best_params_
 #    {'n_estimators': 600,
 # 'min_samples_split': 5,
 # 'min_samples_leaf': 1,
 # 'max_features': 'sqrt',
 # 'max_depth': 60,
 # 'bootstrap': False}


    model_skfold = RandomForestClassifier(n_estimators = 600,min_samples_split = 5, min_samples_leaf= 1,max_features = 'sqrt', max_depth = 60, bootstrap = False  )
    model_skfold.fit(x1, y1)

    # resultados
    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold)
    print("Accuracy: %.2f%%" % (results_skfold.mean() * 100.0))
    # Accuracy: 91.02%

    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold, scoring='average_precision')
    print("Precision: %.2f%%" % (results_skfold.mean() * 100.0))
    # Precision: 9.32%

    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold, scoring='roc_auc')
    print("Roc AUC: %.2f%%" % (results_skfold.mean() * 100.0))
    # ROC_AUC: 57.09%

    # Curva ROC
    metrics.plot_roc_curve(model_skfold, x1, y1)
    plt.savefig(r'images\roc_random_forest\Curva ROC - Rnadom Forest 0.png')
    return


def ml_random_forest(finalDf_0, finalDf_1, finalDf_2):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # define the model
    skfold = StratifiedKFold(n_splits=5)
    rf = RandomForestClassifier(random_state = 42)





    calculate_results_rf(finalDf_0, rf, random_grid, skfold)
    calculate_results_rf(finalDf_1, rf, random_grid, skfold)
    calculate_results_rf(finalDf_2, rf, random_grid, skfold)




    return print("Calculado pelo Random Forest")