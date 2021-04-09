
import matplotlib.pyplot as plt
from scipy.stats import uniform

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics, model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

from sklearn.model_selection import RandomizedSearchCV


def calculate_results_rl(dataframe, logistic, distributions, skfold ):
    x1 = dataframe.drop('target', axis=1).values
    y1 = dataframe['target'].values

    # Busca randomizada pelos hiperparametros da regressão logistica
    clf = RandomizedSearchCV(logistic, distributions, random_state=0)
    search = clf.fit(x1, y1)

    search.best_params_
    # {'C': 2.195254015709299, 'penalty': 'l1'}

    # Houve um erro e não foi aceito a penalidade em l1
    model_skfold = LogisticRegression(C=2.195254015709299, penalty='l2')

    model_skfold.fit(x1, y1)

    # resultados
    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold)
    print("Accuracy: %.2f%%" % (results_skfold.mean() * 100.0))
    # Accuracy: 93.42%

    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold, scoring='average_precision')
    print("Precision: %.2f%%" % (results_skfold.mean() * 100.0))
    # Precision: 21.67%

    results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold, scoring='roc_auc')
    print("Roc AUC: %.2f%%" % (results_skfold.mean() * 100.0))
    # ROC_AUC: 57.09%

    # Curva ROC
    metrics.plot_roc_curve(model_skfold, x1, y1)
    plt.savefig(r'images\roc_logistic_regression\Curva ROC - Regressão Logística Fase 0.png')
    return


def ml_logistic_regression(finalDf_0, finalDf_1, finalDf_2, iterations):

    #Utilizado Stratified Kfold para sepração dos dataset de treinamento e testes, devido ao dataset original ser altamente desbalanceado
    skfold = StratifiedKFold(n_splits=5)
    logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=iterations,random_state=0)
    distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

    calculate_results_rl(finalDf_0, logistic, distributions, skfold)
    calculate_results_rl(finalDf_1, logistic, distributions, skfold)
    calculate_results_rl(finalDf_2, logistic, distributions, skfold)


    return print("Calculado pelo Logistic Regression")