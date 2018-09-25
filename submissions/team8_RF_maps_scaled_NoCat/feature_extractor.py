import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class FeatureExtractor():
    def __init__(self):
        self.scaler = MinMaxScaler()
        return

    def fit(self, X_df, y):
        # you don't have to do anything here
        # - unless you want to use a combined
        # feature extractor/regressor (like deep)
        X_df_new = X_df.get(['windspeed', 'latitude', 'longitude', 
                            'Jday_predictor', 'initial_max_wind',
                            'max_wind_change_12h', 'dist2land'])

        self.scaler.fit(X_df_new)

        return

    def transform(self, X_df):
        
        X_df_num = X_df.get(['windspeed', 'latitude', 'longitude', 
                              'Jday_predictor', 'initial_max_wind',
                              'max_wind_change_12h', 'dist2land'])
        self.scaler.transform(X_df_num)
        X_df_num = X_df_num.fillna(-1)
        XX = X_df_num.values
        
        # reconstruct also some image data
        # (7 2D image per storm instant: z, u, v, sst, slp, hum, vo700 ):
        grid_l = 11  # size of the image is 11 x 11
    
        z_image = np.zeros([grid_l, grid_l, len(X_df)])
        u_image = np.zeros([grid_l, grid_l, len(X_df)])
        v_image = np.zeros([grid_l, grid_l, len(X_df)])
        sst_image = np.zeros([grid_l, grid_l, len(X_df)])
        slp_image = np.zeros([grid_l, grid_l, len(X_df)])
        hum_image = np.zeros([grid_l, grid_l, len(X_df)])
        vo700_image = np.zeros([grid_l, grid_l, len(X_df)])

        for i in range(grid_l):
            for j in range(grid_l):
                z_image[i, j, :] = X_df['z_' + str(i) + '_' + str(j)].values
                u_image[i, j, :] = X_df['u_' + str(i) + '_' + str(j)].values
                v_image[i, j, :] = X_df['v_' + str(i) + '_' + str(j)].values
                sst_image[i, j, :] = X_df['sst_' + str(i) + '_' + str(j)].values
                slp_image[i, j, :] = X_df['slp_' + str(i) + '_' + str(j)].values
                hum_image[i, j, :] = X_df['hum_' + str(i) + '_' + str(j)].values
                vo700_image[i, j, :] = X_df['vo700_' + str(i) + '_' + str(j)].values
        
        z_image = np.transpose(z_image, [2, 0, 1])
        u_image = np.transpose(u_image, [2, 0, 1])
        v_image = np.transpose(v_image, [2, 0, 1])
        sst_image = np.transpose(sst_image, [2, 0, 1])
        slp_image = np.transpose(slp_image, [2, 0, 1])
        hum_image = np.transpose(hum_image, [2, 0, 1])
        vo700_image = np.transpose(vo700_image, [2, 0, 1])

        z_mean = np.mean(z_image, axis=(1, 2))
        u_mean = np.mean(u_image, axis=(1, 2))
        v_mean = np.mean(v_image, axis=(1, 2))
        sst_mean = np.mean(sst_image, axis=(1, 2))
        slp_mean = np.mean(slp_image, axis=(1, 2))
        hum_mean = np.mean(hum_image, axis=(1, 2))
        vo700_mean = np.mean(vo700_image, axis=(1, 2))

        z_center = z_image[:, int(grid_l / 2), int(grid_l / 2)]
        u_center = u_image[:, int(grid_l / 2), int(grid_l / 2)]
        v_center = v_image[:, int(grid_l / 2), int(grid_l / 2)]
        sst_center = sst_image[:, int(grid_l / 2), int(grid_l / 2)]
        slp_center = slp_image[:, int(grid_l / 2), int(grid_l / 2)]
        hum_center = hum_image[:, int(grid_l / 2), int(grid_l / 2)]
        vo700_center = vo700_image[:, int(grid_l / 2), int(grid_l / 2)]

        # add columns to the feature matrix
        XX = np.insert(XX, len(XX[0]), z_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), z_center, axis=1)
        XX = np.insert(XX, len(XX[0]), u_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), u_center, axis=1)
        XX = np.insert(XX, len(XX[0]), v_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), v_center, axis=1)
        XX = np.insert(XX, len(XX[0]), sst_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), sst_center, axis=1)
        XX = np.insert(XX, len(XX[0]), slp_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), slp_center, axis=1)
        XX = np.insert(XX, len(XX[0]), hum_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), hum_center, axis=1)
        XX = np.insert(XX, len(XX[0]), vo700_mean, axis=1)
        XX = np.insert(XX, len(XX[0]), vo700_center, axis=1)

        return XX
