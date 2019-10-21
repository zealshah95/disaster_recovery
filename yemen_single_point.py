import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime
import geopandas as gpd

def extract_files(path):
    filelist = os.listdir(path)
    # for filename in filelist:
    #     df = pd.read_csv(path + filelist[0])
    df = pd.read_csv(path + 'sanaa_grid_10_27.csv')
    return df

def preprocess_df(df, dz):
    #------Parse date time--------------
    df['date_c'] = df['Agg_Name'].apply(lambda x: datetime.datetime.strptime(x.split('_')[1].split('d')[1], "%Y%m%d").date())
    df['start_time'] = df['Agg_Name'].apply(lambda x: datetime.datetime.strptime(x.split('_')[2].split('t')[1], "%H%M%S%f").time()) #%f is for microseconds
    df['end_time'] = df['Agg_Name'].apply(lambda x: datetime.datetime.strptime(x.split('_')[3].split('e')[1], "%H%M%S%f").time())
    #------Extract bit flag values------
    df['Vflag_bin'] = df['QF_Vflag'].apply(lambda x: '{:032b}'.format(x))
    # flip the array of bits: from left->right to right-> left
    # because binary numbers are indexed from right to left
    ### NOTE: another approach would be to use BITWISE AND
    df['Vflag_bin'] = df['Vflag_bin'].apply(lambda x: x[::-1])
    # extract flags corresponding to each metric using standard python indexing
    df['flag_cloud'] = df['Vflag_bin'].apply(lambda x: x[3:5])
    df['lunar_illum'] = df['Vflag_bin'].apply(lambda x: x[5])
    df['day_night_term'] = df['Vflag_bin'].apply(lambda x: x[6:8])
    df['fire_detect'] = df['Vflag_bin'].apply(lambda x: x[8:14])
    df['stray_light'] = df['Vflag_bin'].apply(lambda x: x[14:16])
    df['cloud2_rej'] = df['Vflag_bin'].apply(lambda x: x[18])
    df['dnb_light'] = df['Vflag_bin'].apply(lambda x: x[22:24])
    df['dnb_saa'] = df['Vflag_bin'].apply(lambda x: x[24])
    df['no_data'] = df['Vflag_bin'].apply(lambda x: x[31])
    #------Add Satellite Zenith Angle------
    df = pd.merge(df, dz, on='Sample_DNB')
    #-----Filter out cloud covered and lunar illum data points-----
    dz = df[(df.flag_cloud == '00') & (df.lunar_illum == '1')]
    return df, dz

def create_plots(df, x, y):
    # No cloud cover
    d_nc = df[(df.flag_cloud == '00')]
    # zero lunar illuminance
    d_zli = df[(df.lunar_illum == '1')]
    # no clouds and zero lunar
    d_comb = df[(df.flag_cloud == '00') & (df.lunar_illum == '1')]
    # create array of three datasets
    fig, axs = plt.subplots(1,4)
    axs[0].scatter(df[x], df[y], s=2)
    axs[0].set_title('All Observations')
    axs[1].scatter(d_nc[x], d_nc[y], s=2)
    axs[1].set_title('Clear Sky')
    axs[2].scatter(d_zli[x], d_zli[y], s=2)
    axs[2].set_title('Zero Lunar Illuminance')
    axs[3].scatter(d_comb[x], d_comb[y], s=2)
    axs[3].set_title('Clear Sky/Zero Lunar Illuminance')
    fig.suptitle('{}'.format(df.id[0]))
    for i in range(4):
        axs[i].set_ylim([df[y].min(), df[y].max()])
    plt.show()
    return None

if __name__=='__main__':
    CURRENTDIR = os.getcwd()
    re = CURRENTDIR + '/sanaa_csv/'
    dg = extract_files(re)
    dz = pd.read_excel(CURRENTDIR + '/sanaa_sat_zenith/sanaa_sat_zen.xlsx')
    do, df = preprocess_df(dg.copy(), dz.copy())
    # create_plots(do.copy(), x = 'date_c', y = 'RadE9_DNB')
    # create_plots(do.copy(), x = 'SatZ', y = 'RadE9_DNB')
    code.interact(local = locals())