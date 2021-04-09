
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA



def pca_transform_analysis(scaled_array_p0, scaled_array_p1, scaled_array_p2):

    # Devido ao grande numero de atributos ou dimensoes, decidiu-se fazer uma redução de dimensionalidade por PCA
    pca_phase0 = PCA().fit(scaled_array_p0)
    plt.plot(np.cumsum(pca_phase0.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.savefig(r'images\pca\PCA_PHASE0.png')

    pca_phase1 = PCA().fit(scaled_array_p1)
    plt.plot(np.cumsum(pca_phase1.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.savefig(r'images\pca\PCA_PHASE1.png')

    pca_phase2 = PCA().fit(scaled_array_p2)
    plt.plot(np.cumsum(pca_phase2.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.savefig(r'images\pca\PCA_PHASE2.png')

    # O resultado mostrou que 2 componentes já representam a maior variabilidade dos 3 subdatasets
    pca = PCA(n_components=2)
    principalComponents_phase0 = pca.fit_transform(scaled_array_p0)
    principalComponents_phase1 = pca.fit_transform(scaled_array_p1)
    principalComponents_phase2 = pca.fit_transform(scaled_array_p2)

    # Converte os arrays em datafrme
    principalDf_phase0 = pd.DataFrame(data=principalComponents_phase0)
    principalDf_phase1 = pd.DataFrame(data=principalComponents_phase1)
    principalDf_phase2 = pd.DataFrame(data=principalComponents_phase2)

    return principalDf_phase0, principalDf_phase1, principalDf_phase2
