
import pandas as pd


def data_preparation_ml(df_0_only_target,df_1_only_target, df_2_only_target, principalDf_phase0, principalDf_phase1, principalDf_phase2):
    #concatena para a tabela final normalizada  e com targets
    df_0_only_target.reset_index(drop=True, inplace=True)
    df_1_only_target.reset_index(drop=True, inplace=True)
    df_2_only_target.reset_index(drop=True, inplace=True)
    finalDf_0 = pd.concat([principalDf_phase0, df_0_only_target], axis = 1, ignore_index = True)
    finalDf_1 = pd.concat([principalDf_phase1, df_1_only_target], axis = 1, ignore_index = True)
    finalDf_2 = pd.concat([principalDf_phase2, df_2_only_target], axis = 1, ignore_index = True)
    finalDf_0 = finalDf_0.rename(columns={2: "target"})
    finalDf_1 = finalDf_1.rename(columns={2: "target"})
    finalDf_2 = finalDf_2.rename(columns={2: "target"})


    return finalDf_0, finalDf_1 , finalDf_2

def data_preparation_ml_without_pca(df_0_only_target,df_1_only_target, df_2_only_target, principalDf_phase0, principalDf_phase1, principalDf_phase2):
    #concatena para a tabela final normalizada  e com targets
    principalDf_phase0 = pd.DataFrame(data=principalDf_phase0)
    principalDf_phase1 = pd.DataFrame(data=principalDf_phase1)
    principalDf_phase2 = pd.DataFrame(data=principalDf_phase2)
    df_0_only_target.reset_index(drop=True, inplace=True)
    df_1_only_target.reset_index(drop=True, inplace=True)
    df_2_only_target.reset_index(drop=True, inplace=True)
    finalDf_0 = pd.concat([principalDf_phase0, df_0_only_target], axis=1, ignore_index=True)
    finalDf_1 = pd.concat([principalDf_phase1, df_1_only_target], axis=1, ignore_index=True)
    finalDf_2 = pd.concat([principalDf_phase2, df_2_only_target], axis=1, ignore_index=True)
    finalDf_0 = finalDf_0.rename(columns={2400: "target"})
    finalDf_1 = finalDf_1.rename(columns={2400: "target"})
    finalDf_2 = finalDf_2.rename(columns={2400: "target"})

    return finalDf_0, finalDf_1 , finalDf_2