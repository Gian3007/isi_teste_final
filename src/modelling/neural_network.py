import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout

def neural_network_setup(scaled_array_p0, scaled_array_p1, scaled_array_p2,df_0_only_target,df_1_only_target, df_2_only_target):
    model = Sequential([
        Dense(units=20, input_dim=scaled_array_p0.shape[1], activation='relu'),
        Dense(units=24, activation='relu'),
        Dropout(0.5),
        Dense(units=20, activation='relu'),
        Dense(units=24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    oversample = SMOTE()

    X_smote, y_smote = oversample.fit_resample(scaled_array_p0, df_0_only_target)
    X_smote = pd.DataFrame(X_smote)
    y_smote = pd.DataFrame(y_smote)


    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=0)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=30, epochs=5)
    score = model.evaluate(X_test, y_test)
    print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(score[1] * 100, score[0]))
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    y_pred2 = model.predict(scaled_array_p0)
    y_test2 = pd.DataFrame(df_0_only_target)

    scoreNew = model.evaluate(scaled_array_p0, df_0_only_target)
    print('Test Accuracy: {:.2f}%\nTest Loss: {}'.format(scoreNew[1] * 100, scoreNew[0]))
    print(classification_report(y_test2, y_pred2.round()))

    return print("Rede Neural calculada")
