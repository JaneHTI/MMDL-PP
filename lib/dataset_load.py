import torch
import h5py
import numpy as np
import os
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


DEBUG_ON = 0


# ---------- internal ---------- #
def load_abcd_clinic_data(df, sub_num):
    # clinical variables
    clinic_array = df.iloc[:sub_num, 1:].values
    clinic_tensor = torch.tensor(clinic_array, dtype=torch.float32)

    if DEBUG_ON == 1:
        subids = df.iloc[:sub_num, 0].values
        print('clinic subids: ', subids)

    # continuous risk score -> binary risk label
    score_array = df.iloc[:sub_num, -1].values
    label_array = np.where(score_array >= 65, 1, 0)
    label_tensor = torch.tensor(label_array, dtype=torch.float32)

    return clinic_tensor, label_tensor

def load_abcd_mri_data(t1_mat_dir, sc_h5_dir, fc_h5_dir, sub_num):
    # T1
    t1_mat = sio.loadmat(t1_mat_dir)
    thick_array = t1_mat['thick'][:sub_num, :]  # (N,68)
    area_array = t1_mat['area'][:sub_num, :]  # (N,68)
    volume_array = t1_mat['volume'][:sub_num, :]  # (N,68)
    sub_volume_array = t1_mat['sub_volume'][:sub_num, :]  # (N,16)

    if DEBUG_ON == 1:
        t1_subids = t1_mat['sub_ids'][:sub_num, :]  # (N,1)
        print('t1_subids: ', t1_subids)

    if np.isnan(thick_array).any():
        print("Thick contains NaN values.")

    if np.isnan(area_array).any():
        print("Area contains NaN values.")

    if np.isnan(volume_array).any():
        print("Volume contains NaN values.")

    if np.isnan(sub_volume_array).any():
        print("Sub_volume contains NaN values.")

    ## z-score
    scaler = StandardScaler()
    thick_array_scaled = scaler.fit_transform(thick_array)
    area_array_scaled = scaler.fit_transform(area_array)
    volume_array_scaled = scaler.fit_transform(volume_array)
    sub_volume_array_scaled = scaler.fit_transform(sub_volume_array)

    ## array -> tensor
    thick_tensor = torch.tensor(thick_array_scaled, dtype=torch.float32)  # [N,68]
    area_tensor = torch.tensor(area_array_scaled, dtype=torch.float32)  # [N,68]
    volume_tensor = torch.tensor(volume_array_scaled, dtype=torch.float32)  # [N,68]
    sub_volume_tensor = torch.tensor(sub_volume_array_scaled, dtype=torch.float32)  # [N,16]

    # SC
    with h5py.File(sc_h5_dir, 'r') as sc_h5f:
        # print("SC:", list(sc_h5f.keys()))
        sc_array = sc_h5f['sc'][:sub_num, :, :]  # (N,268,268)

        if DEBUG_ON == 1:
            sc_subids = sc_h5f['sub_ids'][:sub_num, :]  # (N,1)
            print('sc_subids: ', sc_subids)

        ## set the diagonal elements to 0
        for i in range(sc_array.shape[0]):
            np.fill_diagonal(sc_array[i], 0) ####

        ## nan
        if np.isnan(sc_array).any():
            sc_array = np.nan_to_num(sc_array, nan=0.0)

        if np.isnan(sc_array).any():
            print("SC contains NaN values.")

        ## array -> tensor
        sc_tensor = torch.tensor(sc_array, dtype=torch.float32)  # [N,268,268]

    # FC
    with h5py.File(fc_h5_dir, 'r') as fc_h5f:
        # print("FC:", list(fc_h5f.keys()))
        fc_array = fc_h5f['fc'][:sub_num, :, :]  # (N,268,268)

        if DEBUG_ON == 1:
            fc_subids = fc_h5f['sub_ids'][:sub_num, :]  # (N,1)
            print('fc_subids: ', fc_subids)

        ## set the diagonal elements to 0
        for i in range(fc_array.shape[0]):
            np.fill_diagonal(fc_array[i], 0)   ####

        ## nan
        if np.isnan(fc_array).any():
            fc_array = np.nan_to_num(fc_array, nan=0.0)

        if np.isnan(fc_array).any():
            print("FC contains NaN values.")

        ## array -> tensor
        fc_tensor = torch.tensor(fc_array, dtype=torch.float32)  # [N,268,268]

    return thick_tensor, area_tensor, volume_tensor, sub_volume_tensor, sc_tensor, fc_tensor


# ---------- external ---------- #
def load_train_test_data(data_name, sub_name, rand_seed):
    print(f'# ------------------------ load_train_test_data ---------------------- #')
    root_dir = os.path.join(script_dir, '..', 'data/')

    if data_name == 'ABCD':
        print('# ------------------------ Loading ABCD.')
        abcd_clinic_root_dir = root_dir + 'ABCD/'
        abcd_mri_root_dir = root_dir + 'ABCD/'

        if sub_name == 'ABCD_YR0':
            print('# ------------------------ Loading ABCD_YR0.')
            # YR0
            ## clinic variables and risk score
            abcd_y0_clinic_dir = abcd_clinic_root_dir + 'abcd_y0_clinical_variables_norm_8705.xlsx'
            df_abcd_y0_clinic = pd.read_excel(abcd_y0_clinic_dir)
            abcd_y0_sub_num = df_abcd_y0_clinic.shape[0] - 1
            abcd_y0_clinic_tensor, abcd_y0_label_tensor = load_abcd_clinic_data(df_abcd_y0_clinic, abcd_y0_sub_num)
            print('abcd_y0_clinic_tensor:', abcd_y0_clinic_tensor.shape)
            print('abcd_y0_label_tensor:', abcd_y0_label_tensor.shape)

            abcd_y0_risk_n = torch.sum(abcd_y0_label_tensor >= 1).item()
            abcd_y0_risk_r = abcd_y0_risk_n / len(abcd_y0_label_tensor)
            print(f'abcd_y0_risk: {abcd_y0_risk_n:.0f}, {abcd_y0_risk_r:.4f}')

            ## MRI features
            abcd_y0_t1_dir = abcd_mri_root_dir + 'abcd_wave1_t1_8705.mat'
            abcd_y0_sc_dir = abcd_mri_root_dir + 'abcd_wave1_sc_8705.h5'
            abcd_y0_fc_dir = abcd_mri_root_dir + 'abcd_wave1_fc_8705.h5'

            (abcd_y0_ct_tensor,
             abcd_y0_ca_tensor,
             abcd_y0_cv_tensor,
             abcd_y0_sv_tensor,
             abcd_y0_sc_tensor,
             abcd_y0_fc_tensor) = load_abcd_mri_data(abcd_y0_t1_dir,
                                                     abcd_y0_sc_dir,
                                                     abcd_y0_fc_dir,
                                                     abcd_y0_sub_num)
            print('abcd_y0_ct_tensor:', abcd_y0_ct_tensor.shape)
            print('abcd_y0_ca_tensor:', abcd_y0_ca_tensor.shape)
            print('abcd_y0_cv_tensor:', abcd_y0_cv_tensor.shape)
            print('abcd_y0_sv_tensor:', abcd_y0_sv_tensor.shape)
            print('abcd_y0_sc_tensor:', abcd_y0_sc_tensor.shape)
            print('abcd_y0_fc_tensor:', abcd_y0_fc_tensor.shape)

            # divide ABCD YR0 into train and test
            train_size = 7 / 10
            df_abcd_y0_train_clinic, df_abcd_y0_test_clinic = train_test_split(df_abcd_y0_clinic.iloc[:abcd_y0_sub_num],
                                                                               train_size=train_size,
                                                                               random_state=rand_seed,
                                                                               shuffle=True,
                                                                               stratify=abcd_y0_label_tensor.numpy())

            abcd_y0_train_sub_num = df_abcd_y0_train_clinic.shape[0]
            train_clinic_tensor, train_label_tensor = load_abcd_clinic_data(df_abcd_y0_train_clinic,
                                                                            abcd_y0_train_sub_num)
            print('abcd_y0_train_clinic_tensor:', train_clinic_tensor.shape)
            print('abcd_y0_train_label_tensor:', train_label_tensor.shape)

            abcd_y0_test_sub_num = df_abcd_y0_test_clinic.shape[0]
            test_clinic_tensor, test_label_tensor = load_abcd_clinic_data(df_abcd_y0_test_clinic,
                                                                          abcd_y0_test_sub_num)
            print('abcd_y0_test_clinic_tensor:', test_clinic_tensor.shape)
            print('abcd_y0_test_label_tensor:', test_label_tensor.shape)

            ## MRI features
            train_indices = df_abcd_y0_train_clinic.index.tolist()
            test_indices = df_abcd_y0_test_clinic.index.tolist()

            train_ct_tensor = abcd_y0_ct_tensor[train_indices]
            train_ca_tensor = abcd_y0_ca_tensor[train_indices]
            train_cv_tensor = abcd_y0_cv_tensor[train_indices]
            train_sv_tensor = abcd_y0_sv_tensor[train_indices]
            train_sc_tensor = abcd_y0_sc_tensor[train_indices]
            train_fc_tensor = abcd_y0_fc_tensor[train_indices]
            print('abcd_y0_train_ct_tensor:', train_ct_tensor.shape)
            print('abcd_y0_train_ca_tensor:', train_ca_tensor.shape)
            print('abcd_y0_train_cv_tensor:', train_cv_tensor.shape)
            print('abcd_y0_train_sv_tensor:', train_sv_tensor.shape)
            print('abcd_y0_train_sc_tensor:', train_sc_tensor.shape)
            print('abcd_y0_train_fc_tensor:', train_fc_tensor.shape)

            test_ct_tensor = abcd_y0_ct_tensor[test_indices]
            test_ca_tensor = abcd_y0_ca_tensor[test_indices]
            test_cv_tensor = abcd_y0_cv_tensor[test_indices]
            test_sv_tensor = abcd_y0_sv_tensor[test_indices]
            test_sc_tensor = abcd_y0_sc_tensor[test_indices]
            test_fc_tensor = abcd_y0_fc_tensor[test_indices]
            print('abcd_y0_test_ct_tensor:', test_ct_tensor.shape)
            print('abcd_y0_test_ca_tensor:', test_ca_tensor.shape)
            print('abcd_y0_test_cv_tensor:', test_cv_tensor.shape)
            print('abcd_y0_test_sv_tensor:', test_sv_tensor.shape)
            print('abcd_y0_test_sc_tensor:', test_sc_tensor.shape)
            print('abcd_y0_test_fc_tensor:', test_fc_tensor.shape)

    # calculate train_risk_r and test_risk_r
    train_risk_n = torch.sum(train_label_tensor >= 1).item()
    train_risk_r = train_risk_n / len(train_label_tensor)
    print(f'train_risk: {train_risk_n:.0f}, {train_risk_r:.4f}')

    test_risk_n = torch.sum(test_label_tensor >= 1).item()
    test_risk_r = test_risk_n / len(test_label_tensor)
    print(f'test_risk: {test_risk_n:.0f}, {test_risk_r:.4f}')

    train_set = {
        'train_label_tensor': train_label_tensor,
        'train_clinic_tensor': train_clinic_tensor,
        'train_ct_tensor': train_ct_tensor,
        'train_ca_tensor': train_ca_tensor,
        'train_cv_tensor': train_cv_tensor,
        'train_sv_tensor': train_sv_tensor,
        'train_sc_tensor': train_sc_tensor,
        'train_fc_tensor': train_fc_tensor,
        'train_risk_r': train_risk_r
    }

    test_set = {
        'test_label_tensor': test_label_tensor,
        'test_clinic_tensor': test_clinic_tensor,
        'test_ct_tensor': test_ct_tensor,
        'test_ca_tensor': test_ca_tensor,
        'test_cv_tensor': test_cv_tensor,
        'test_sv_tensor': test_sv_tensor,
        'test_sc_tensor': test_sc_tensor,
        'test_fc_tensor': test_fc_tensor,
        'test_risk_r': test_risk_r
    }

    return train_set, test_set

def load_entire_data(data_name, sub_name):
    print(f'# ------------------------ load_entire_data ---------------------- #')
    root_dir = os.path.join(script_dir, '..', 'data/')

    if data_name == 'ABCD':
        abcd_clinic_root_dir = root_dir + 'ABCD/'
        abcd_mri_root_dir = root_dir + 'ABCD/'

        if sub_name == 'ABCD_demo':
            print('# ------------------------ Loading ABCD_demo.')
            ## clinic variables and risk score
            abcd_demo_clinic_dir = abcd_clinic_root_dir + 'abcd_demo_clinic.xlsx'
            df_abcd_demo_clinic = pd.read_excel(abcd_demo_clinic_dir)
            abcd_demo_sub_num = df_abcd_demo_clinic.shape[0]
            clinic_tensor, label_tensor = load_abcd_clinic_data(df_abcd_demo_clinic, abcd_demo_sub_num)
            print('abcd_demo_clinic_tensor:', clinic_tensor.shape)
            print('abcd_demo_label_tensor:', label_tensor.shape)

            ## MRI features
            abcd_demo_t1_dir = abcd_mri_root_dir + 'abcd_demo_t1.mat'
            abcd_demo_sc_dir = abcd_mri_root_dir + 'abcd_demo_sc.h5'
            abcd_demo_fc_dir = abcd_mri_root_dir + 'abcd_demo_fc.h5'

            (ct_tensor,
             ca_tensor,
             cv_tensor,
             sv_tensor,
             sc_tensor,
             fc_tensor) = load_abcd_mri_data(abcd_demo_t1_dir,
                                             abcd_demo_sc_dir,
                                             abcd_demo_fc_dir,
                                             abcd_demo_sub_num)
            print('abcd_demo_ct_tensor:', ct_tensor.shape)
            print('abcd_demo_ca_tensor:', ca_tensor.shape)
            print('abcd_demo_cv_tensor:', cv_tensor.shape)
            print('abcd_demo_sv_tensor:', sv_tensor.shape)
            print('abcd_demo_sc_tensor:', sc_tensor.shape)
            print('abcd_demo_fc_tensor:', fc_tensor.shape)

    # calculate train_risk_r and test_risk_r
    risk_n = torch.sum(label_tensor >= 1).item()
    risk_r = risk_n / len(label_tensor)
    print(f'risk: {risk_n:.0f}, {risk_r:.4f}')

    data_set = {
        'label_tensor': label_tensor,
        'clinic_tensor': clinic_tensor,
        'ct_tensor': ct_tensor,
        'ca_tensor': ca_tensor,
        'cv_tensor': cv_tensor,
        'sv_tensor': sv_tensor,
        'sc_tensor': sc_tensor,
        'fc_tensor': fc_tensor,
        'risk_r': risk_r
    }
    return data_set