import pandas as pd
import numpy as np
 
class FeatureExtractor(object):
    def __init__(self):
        self.spatial_fields = ["u", "v", "sst", "slp", "hum","vo700","z"]
        self.scaling_values = pd.DataFrame(index=self.spatial_fields, 
                                           columns=["mean", "std"], dtype=float)

    def fit(self, X_df, y): 
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11) 
            field_grids.append(f_data)
        for f, field in enumerate(self.spatial_fields):
            self.scaling_values.loc[field, "mean"] = np.nanmean(field_grids[f])
            self.scaling_values.loc[field, "std"] = np.nanstd(field_grids[f])
        print(self.scaling_values)
 
    def transform(self, X_df):
        field_grids = []
        for field in self.spatial_fields:
            f_cols = X_df.columns[X_df.columns.str.contains(field + "_")]
            f_data = X_df[f_cols].values.reshape(-1, 11, 11) 
            field_grids.append((f_data - self.scaling_values.loc[field, "mean"]) / self.scaling_values.loc[field, "std"])
            field_grids[-1][np.isnan(field_grids[-1])] = 0 
        norm_data = np.stack(field_grids, axis=-1)

        return norm_data