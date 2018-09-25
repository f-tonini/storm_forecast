import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureExtractor(object):
    def __init__(self):
        self.scalar_fields = ["latitude", "longitude", "windspeed", "instant_t", "dist2land"]
        self.spatial_fields = ["u", "v", "sst", "slp", "hum"]
        self.scaling_values = pd.DataFrame(index=self.spatial_fields, 
                                           columns=["mean", "std"], dtype=float)
        self.scalar_norm = StandardScaler()

    def fit(self, X_df, y): 
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11) 
            field_grids.append(f_data)
        for f, field in enumerate(self.spatial_fields):
            self.scaling_values.loc[field, "mean"] = np.nanmean(field_grids[f])
            self.scaling_values.loc[field, "std"] = np.nanstd(field_grids[f])
        self.scalar_norm.fit(X_df[self.scalar_fields])

    def transform(self, X_df):
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11) 
            field_grids.append((f_data - self.scaling_values.loc[field, "mean"]) / self.scaling_values.loc[field, "std"])
            field_grids[-1][np.isnan(field_grids[-1])] = 0 
        norm_data = np.stack(field_grids, axis=-1)
        norm_scalar = self.scalar_norm.transform(X_df[self.scalar_fields])
        return [norm_data, norm_scalar]
