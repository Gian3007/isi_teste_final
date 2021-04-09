from sklearn.preprocessing import StandardScaler


def normalize_data(df_0_without_nan_only_features,df_1_without_nan_only_features, df_2_without_nan_only_features ):
    sc = StandardScaler()
    scaled_array_p0 = sc.fit_transform(df_0_without_nan_only_features)
    scaled_array_p1 = sc.fit_transform(df_1_without_nan_only_features)
    scaled_array_p2 = sc.fit_transform(df_2_without_nan_only_features)

    return scaled_array_p0, scaled_array_p1, scaled_array_p2