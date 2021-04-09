from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn import model_selection
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, metrics, model_selection, svm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


from modelling import logistic_regression, random_forest, neural_network
from utils import data_preparation, pca_analysis, standard_scaler

alghoritm =3

#Gera 2400 amostras randomicas numa população de 799999

seed =45
randomlist = random.sample(range(0, 799999), 2400)


data_measurements = pd.read_csv(r'repository\database.csv', usecols=randomlist)
#dataset de features numericas, ou seja, as medições
phase_target = pd.read_csv(r'repository\database.csv', usecols=['phase','target'])
#dataset de features das categorias

phase_target.info()
#sem NaNs, int64, 501 linhas

data_measurements.info()
#sem NaNs, int64, 501 linhas, 2400 colunas

data_measurements.describe()
#dados estatísticos das features do dataset




#Concatenção dos dataframes e depois separação em 3 dataframes, uma para cada fase das linahs de transmissão
full_dataframe = pd.concat([data_measurements, phase_target.reindex(data_measurements.index)], axis=1)
df_0 = full_dataframe[full_dataframe.phase ==0]
df_1 = full_dataframe[full_dataframe.phase ==1]
df_2 = full_dataframe[full_dataframe.phase ==2]

#exclui qualquer coluna com possiveis valores Nan
df_0_without_nan = df_0.dropna(axis =1)
df_1_without_nan = df_1.dropna(axis =1)
df_2_without_nan = df_2.dropna(axis =1)

#Separação de dataframes de features e target
df_0_without_nan_only_features = df_0_without_nan.drop(columns =['phase','target'])
df_0_only_target = df_0_without_nan[['target']]
print ("Número de anomalias na Fase 0", df_0_only_target[df_0_only_target == 1].sum())
df_1_without_nan_only_features = df_1_without_nan.drop(columns =['phase','target'])
df_1_only_target = df_1_without_nan[['target']]
print ("Número de anomalias na Fase 1", df_1_only_target[df_1_only_target == 1].sum())
df_2_without_nan_only_features = df_2_without_nan.drop(columns =['phase','target'])
df_2_only_target = df_2_without_nan[['target']]
print ("Número de anomalias na Fase 2", df_2_only_target[df_2_only_target == 1].sum())



########################################################################################################################
scaled_array_p0, scaled_array_p1, scaled_array_p2 = standard_scaler.normalize_data(df_0_without_nan_only_features,df_1_without_nan_only_features, df_2_without_nan_only_features )
principalDf_phase0, principalDf_phase1, principalDf_phase2 = pca_analysis.pca_transform_analysis(scaled_array_p0, scaled_array_p1, scaled_array_p2)
finalDf_0, finalDf_1 , finalDf_2 = data_preparation.data_preparation_ml(df_0_only_target,df_1_only_target, df_2_only_target, principalDf_phase0, principalDf_phase1, principalDf_phase2)
logistic_regression.ml_logistic_regression(finalDf_0, finalDf_1, finalDf_2, iterations  = 250)

########################################################################################################################
scaled_array_p0, scaled_array_p1, scaled_array_p2 = standard_scaler.normalize_data(df_0_without_nan_only_features,df_1_without_nan_only_features, df_2_without_nan_only_features )
principalDf_phase0, principalDf_phase1, principalDf_phase2 = pca_analysis.pca_transform_analysis(scaled_array_p0, scaled_array_p1, scaled_array_p2)
finalDf_0, finalDf_1 , finalDf_2 = data_preparation.data_preparation_ml(df_0_only_target,df_1_only_target, df_2_only_target, principalDf_phase0, principalDf_phase1, principalDf_phase2)
random_forest.ml_random_forest(finalDf_0, finalDf_1, finalDf_2)

########################################################################################################################
scaled_array_p0, scaled_array_p1, scaled_array_p2 = standard_scaler.normalize_data(df_0_without_nan_only_features,df_1_without_nan_only_features, df_2_without_nan_only_features )
neural_network.neural_network_setup(scaled_array_p0, scaled_array_p1, scaled_array_p2,df_0_only_target,df_1_only_target, df_2_only_target)




if __name__ == "__main__":
    print("Start")