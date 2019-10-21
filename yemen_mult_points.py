import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import code
import datetime
import geopandas as gpd
from sklearn import linear_model

def create_daily_database(path):
    filelist = os.listdir(path)
    df_to_append = []
    for filename in filelist:
        df = pd.read_csv(path + filename)
        db = preprocess_df(df)
        # db = d_filt.groupby(['id', 'Latitude', 'Longitude', 'date_c']).mean().reset_index() #removes duplicates
        db['day'] = db.date_c.apply(lambda x: x.day)
        db['month'] = db.date_c.apply(lambda x: x.month)
        db['year'] = db.date_c.apply(lambda x: x.year)
        df_to_append.append(db)
    apd = pd.concat(df_to_append)
    code.interact(local = locals())
    return appended_data

def preprocess_df(df):
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
    # df = pd.merge(df, dz, on='Sample_DNB')
    #-----Filter out cloud covered and lunar illum data points-----
    df = df[(df.flag_cloud == '00') & (df.lunar_illum == '1')]
    # Output filtered and unfiltered dataframes
    return df

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

def recovery_rate(df, timeframe):
    df.date_c = pd.to_datetime(df.date_c)
    df = df[(df.date_c > pd.Timestamp(2015, 3, 31))]
    if timeframe == 'monthly':
        dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    elif timeframe == 'yearly':
        dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('Y', on='date_c').mean()[['RadE9_DNB']].reset_index()
    else:
        dg = df.copy()
    dg['date_ordinal'] = dg['date_c'].map(datetime.datetime.toordinal)
    ds = dg[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    slope = []
    for s in dg.sort_values(by=['id']).id.unique():
        dh = dg[(dg.id == s)]
        reg = linear_model.LinearRegression()
        reg.fit(dh['date_ordinal'].values.reshape(-1,1), dh['RadE9_DNB'].values)
        # rad_pred = reg.predict(dh['date_ordinal'].values.reshape(-1,1))
        if timeframe == "monthly":
            m = 30.5*reg.coef_ # average of 30 and 31
        elif timeframe == "yearly":
            m = 365*reg.coef_
        else:
            m = reg.coef_
        slope.append(m[0])
        print('{}: {} change of {} radiance'.format(s, timeframe, m))
    ds['{}_rate'.format(timeframe)] = slope
    return ds

def radiance_change(df):
    df.date_c = pd.to_datetime(df.date_c)
    # average radiance value recorded during bombing month and during latest month
    dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dp = dm[(dm.date_c.dt.month == 4) & (dm.date_c.dt.year == 2015)] # bombing on March 26, 2015 so we take April 2015 full
    dl = dm[(dm.date_c.dt.month == 5) & (dm.date_c.dt.year == 2019)] #last month of dataset is May 2019
    # create empty array to append values for radiance change
    ds = df[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    print("###############################################################################")
    rad_post_event = []
    rad_latest = []
    for s in df.sort_values(by='id').id.unique():
        rad_post_event.append(dp[(dp.id == s)].RadE9_DNB.values[0])
        rad_latest.append(dl[(dl.id == s)].RadE9_DNB.values[0])
        # print("{} - Before: {}, After: {}".format(s, rad_post_event, rad_latest))
    ds['rad_before'] = rad_post_event
    ds['rad_after'] = rad_latest
    ds['rad_p'] = (ds['rad_after'] - ds['rad_before'])*100.0/ds['rad_before']
    return ds

def affected_areas(df):
    df.date_c = pd.to_datetime(df.date_c)
    df = df.drop(df.index[1817394])
    # average radiance value recorded during bombing month and during latest month
    dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dp = dm[(dm.date_c.dt.month == 2) & (dm.date_c.dt.year == 2015)] # bombing on March 26, 2015 so we take April 2015 full
    dl = dm[(dm.date_c.dt.month == 4) & (dm.date_c.dt.year == 2015)] # last month of dataset is May 2019
    # create empty array to append values for radiance change
    ds = df[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    print("###############################################################################")
    rad_post_event = []
    rad_latest = []
    for s in df.sort_values(by='id').id.unique():
        rad_post_event.append(dp[(dp.id == s)].RadE9_DNB.values[0])
        rad_latest.append(dl[(dl.id == s)].RadE9_DNB.values[0])
        # print("{} - Before: {}, After: {}".format(s, rad_post_event, rad_latest))
    ds['rad_before'] = rad_post_event
    ds['rad_after'] = rad_latest
    ds['rad_p'] = (ds['rad_after'] - ds['rad_before'])*100.0/ds['rad_before']
    return ds

def updated_recovery(df):
    df.date_c = pd.to_datetime(df.date_c)
    # average radiance value recorded during bombing month and during latest month
    dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dp = dm[(dm.date_c.dt.month.isin([1,2,3])) & (dm.date_c.dt.year == 2015)] # right before bombing - Jan,Feb,Mar 2015
    dp = dp.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()

    dl = dm[(dm.date_c.dt.month == 5) & (dm.date_c.dt.year == 2019)] #last month of dataset is May 2019
    # create empty array to append values for radiance change
    ds = df[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    print("###############################################################################")
    rad_pre_event = []
    rad_latest = []
    for s in df.sort_values(by='id').id.unique():
        rad_pre_event.append(dp[(dp.id == s)].RadE9_DNB.values[0])
        rad_latest.append(dl[(dl.id == s)].RadE9_DNB.values[0])
        # print("{} - Before: {}, After: {}".format(s, rad_post_event, rad_latest))
    ds['rad_before'] = rad_pre_event
    ds['rad_after'] = rad_latest
    ds['rad_p'] = (ds['rad_after'])*100.0/ds['rad_before']
    return ds

def bright_centers_comparison(df):
    dh = df.copy()
    #-----Prepare data---------------------
    # drr = recovery_rate(df.copy(), 'daily')
    # df.date_c = pd.to_datetime(df.date_c)
    # df = df.drop(df.index[1817394])
    # dp = df[(df.date_c.dt.month.isin([1,2,3])) & (df.date_c.dt.year == 2015)]
    # dp = dp.groupby(['id', 'Latitude', 'Longitude']).agg({'RadE9_DNB':['mean', 'median', 'max']}).reset_index()
    # dp.columns = [' '.join(col).strip() for col in dp.columns.values]
    # dp = dp.rename(columns = {'RadE9_DNB mean':'rad_mean', 'RadE9_DNB median':'rad_med', 'RadE9_DNB max':'rad_max'})
    # dg = pd.merge(dp, drr, on=['id', 'Latitude', 'Longitude'])
    #------Read data-----------------------
    d_af = pd.read_pickle('affected_areas_2')
    # d_af = affected_areas(df.copy())
    dg = pd.read_pickle('centers_comparison_2')
    code.interact(local = locals())
    return None

def histogram_plots(df,topic):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    if topic=="before_blast_radiance":
        dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
        dp = dm[(dm.date_c.dt.month.isin([1,2,3])) & (dm.date_c.dt.year == 2015)] # right before bombing - Jan,Feb,Mar 2015
        dp = dp.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
        plt.hist(dp.RadE9_DNB, 10) # low, moderate, high, very high
        plt.show()
    elif topic == "radiance_today":
        dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
        dl = dm[(dm.date_c.dt.month == 5) & (dm.date_c.dt.year == 2019)] #last month of dataset is May 2019
        plt.hist(dl.RadE9_DNB, 4) # low, moderate, high, very high
        plt.show()
    elif topic == "radiance_right_after_blasts":
        dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
        dl = dm[(dm.date_c.dt.month == 4) & (dm.date_c.dt.year == 2015)]
        plt.hist(dl.RadE9_DNB, 4)
        plt.show()
    elif topic == "radiance_change":
        dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
        db = dm[(dm.date_c.dt.month.isin([1,2,3])) & (dm.date_c.dt.year == 2015)]
        db = db.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
        da = dm[(dm.date_c.dt.month == 4) & (dm.date_c.dt.year == 2015)]
        da = da.rename(columns = {'RadE9_DNB':'rad_after'})
        db = db.rename(columns = {'RadE9_DNB':'rad_before'})
        dg = pd.merge(db,da, left_on='id', right_on='id')
        dg['rad_p'] = (dg['rad_after'] - dg['rad_before'])*100.0/dg['rad_before']
        # code.interact(local = locals())
        # plt.hist(dl.RadE9_DNB, 20)
        # plt.show()
    # code.interact(local = locals())
    return None

def impact_based_classification(df):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    dm = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    db = dm[(dm.date_c.dt.month.isin([1,2,3])) & (dm.date_c.dt.year == 2015)]
    db = db.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
    da = dm[(dm.date_c.dt.month == 4) & (dm.date_c.dt.year == 2015)]
    da = da.rename(columns = {'RadE9_DNB':'rad_after'})
    db = db.rename(columns = {'RadE9_DNB':'rad_before'})
    da = da.drop(columns = ['Latitude','Longitude'])
    dg = pd.merge(db,da, left_on='id', right_on='id')
    dg['rad_p'] = (dg['rad_after'] - dg['rad_before'])*100.0/dg['rad_before']
    dg['rad_p_abs'] = abs(dg['rad_p'])
    dg = dg[['id','Latitude','Longitude','rad_p_abs']]
    dg['class'] = dg.rad_p_abs.apply(lambda x: "Extreme" if x>=95 else "Very High" if 85<=x<95 else "High" if 75<=x<85 else "Moderate" if 60<=x<75 else "Light" if 40<=x<60 else "Low")
    return dg

def pre_event_rad_based_classification(df):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dg = dg[(dg.date_c.dt.month.isin([1,2,3])) & (dg.date_c.dt.year == 2015)]
    dg = dg.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
    dg['block'] = dg.RadE9_DNB.apply(lambda x: "Very Bright" if x>=60 else "Bright" if 40<=x<60 else "Moderate" if 20<=x<40 else "Low" if 3<=x<20 else "Very Low")
    # code.interact(local = locals())
    return dg

def post_event_rad_based_classification(df):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    code.interact(local = locals())
    dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dg = dg[(dg.date_c.dt.month.isin([4])) & (dg.date_c.dt.year == 2015)]
    dg = dg.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
    dg['block'] = dg.RadE9_DNB.apply(lambda x: "Very Bright" if x>=60 else "Bright" if 40<=x<60 else "Moderate" if 20<=x<40 else "Low" if 3<=x<20 else "Very Low")
    # code.interact(local = locals())
    return dg

def present_day_rad_based_classification(df):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dg = dg[(dg.date_c.dt.month.isin([1,2,3])) & (dg.date_c.dt.year == 2019)]
    dg = dg.groupby(['id', 'Latitude', 'Longitude']).mean()[['RadE9_DNB']].reset_index()
    dg['block'] = dg.RadE9_DNB.apply(lambda x: "Very Bright" if x>=60 else "Bright" if 40<=x<60 else "Moderate" if 20<=x<40 else "Low" if 3<=x<20 else "Very Low")
    # code.interact(local = locals())
    return dg

def track_growth_by_damage(df, dc):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    df = df[(df.date_c > pd.Timestamp(2015, 3, 31))]
    dc = dc[['id','class']]
    db = pd.merge(df,dc,left_on='id',right_on='id')
    db = db.groupby('class').resample('1M',on='date_c').mean()[['RadE9_DNB']].reset_index()
    for i in ['Extreme','Very High','High','Moderate','Light','Low']:
        dx = db[(db['class'] == i)]
        plt.plot(dx.date_c,dx.RadE9_DNB,label=i)
    plt.legend()
    plt.xlabel('Monthly Timestamps')
    plt.ylabel('Mean Radiance')
    plt.title("Evolution of mean radiance over time (monthly analysis)")
    plt.show()
    return None

def track_growth_by_pre_bombing(df,dc):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    df = df[(df.date_c > pd.Timestamp(2015, 3, 31))]
    db = pd.merge(df, dc[['id','block']], left_on='id', right_on='id')
    # code.interact(local = locals())
    db = db.groupby('block').resample('1M',on='date_c').mean()[['RadE9_DNB']].reset_index()
    for i in ['Very Bright','Bright','Moderate', 'Low', 'Very Low']:
        dx = db[(db['block'] == i)]
        plt.plot(dx.date_c,dx.RadE9_DNB,label=i)
    plt.legend()
    plt.xlabel('Monthly Timestamps')
    plt.ylabel('Mean Radiance')
    plt.title("Evolution of mean radiance over time (monthly analysis) \n Legends denote pre-bombing brightness groups")
    plt.show()
    # code.interact(local = locals())
    return None

def monthly_recovery_rate_by_damage(df, dc):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    #-----Recovery Rate Post Bombing (Monthly)----------------------------------------------------------------------
    dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dg = dg[(dg.date_c > pd.Timestamp(2015,3,31))]
    ds = dg[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    slope = []
    for s in dg.sort_values(by=['id']).id.unique():
        dh = dg[(dg.id == s)]
        dh = dh.reset_index()
        dh = dh.drop(columns = ['index'])
        reg = linear_model.LinearRegression()
        reg.fit(dh.index.values.reshape(-1,1), dh['RadE9_DNB'].values)
        m = reg.coef_
        slope.append(m[0])
        print('{} change of {} radiance'.format(s, m))
    ds['rec_rate'] = slope
    #------Add Radiance Before Blasts to DF--------------------------------------------------------------------------
    dr = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dr = dr[(dr.date_c.dt.month.isin([1,2,3])) & (dr.date_c.dt.year==2015)].groupby(['id']).mean()[['RadE9_DNB']].reset_index()
    dr = dr.rename(columns = {'RadE9_DNB':'rad_before'})
    ds1 = pd.merge(ds,dr,left_on='id',right_on='id')
    #-----Merge the Impact classes-----------------------------------------------------------------------------------
    ds2 = pd.merge(ds1,dc[['id','class']],left_on='id',right_on='id')
    #-------Plot-----------------------------------------------------------------------------------------------------
    colors = ['black','red','orange','blue','green','yellow']
    a = 0
    for i in ['Extreme','Very High','High','Moderate','Light','Low']:
        dx = ds2[(ds2['class']==i)]
        plt.scatter(dx.rec_rate,dx.rad_before,s=6,c = colors[a],label=i)
        a = a+1
    plt.legend()
    plt.xlabel('Monthly Recovery Rate')
    plt.ylabel('Pre-Bombing Radiance Levels')
    plt.title("Comparison of recovery rates based on prebombing radiance levels")
    # plt.show()
    # code.interact(local = locals())
    return None

def monthly_recovery_rate_by_rad_centers(df):
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    #-----Recovery Rate Post Bombing (Monthly)----------------------------------------------------------------------
    dg = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dg = dg[(dg.date_c > pd.Timestamp(2015,3,31))]
    ds = dg[['id', 'Latitude', 'Longitude']].drop_duplicates().sort_values(by='id')
    slope = []
    for s in dg.sort_values(by=['id']).id.unique():
        dh = dg[(dg.id == s)]
        dh = dh.reset_index()
        dh = dh.drop(columns = ['index'])
        reg = linear_model.LinearRegression()
        reg.fit(dh.index.values.reshape(-1,1), dh['RadE9_DNB'].values)
        m = reg.coef_
        slope.append(m[0])
        print('{} change of {} radiance'.format(s, m))
    ds['rec_rate'] = slope
    #------Add Radiance Before Blasts to DF--------------------------------------------------------------------------
    dr = df.groupby(['id', 'Latitude', 'Longitude']).resample('M', on='date_c').mean()[['RadE9_DNB']].reset_index()
    dr = dr[(dr.date_c.dt.month.isin([1,2,3])) & (dr.date_c.dt.year==2015)].groupby(['id']).mean()[['RadE9_DNB']].reset_index()
    dr = dr.rename(columns = {'RadE9_DNB':'rad_before'})
    dr['block'] = dr.rad_before.apply(lambda x: "Very Bright" if x>=60 else "Bright" if 40<=x<60 else "Moderate" if 20<=x<40 else "Low" if 3<=x<20 else "Very Low")
    ds2 = pd.merge(ds,dr,left_on='id',right_on='id')
    #-------Plot-----------------------------------------------------------------------------------------------------
    colors = ['black','red','orange','blue','yellow']
    a = 0
    for i in ['Very Bright','Bright','Moderate', 'Low', 'Very Low']:
        dx = ds2[(ds2['block']==i)]
        plt.scatter(dx.rec_rate,dx.rad_before,s=6,c = colors[a],label=i)
        a = a+1
    plt.legend()
    plt.xlabel('Monthly Recovery Rate')
    plt.ylabel('Pre-Bombing Radiance Levels')
    plt.title("Comparison of recovery rates based on prebombing radiance levels")
    plt.show()
    # code.interact(local = locals())
    return None

def calculate_damage_recovery(df):
    print("starting calculations!")
    df.date_c = pd.to_datetime(df.date_c)
    dl = df[(df.id == 'sanaa_grid_28_45') & (df.year == 2015) & (df.month == 4)  & (df.day == 17)] #outlier
    df = df.drop(dl.index)
    print("cleaning done")
    db = df[(df.date_c.dt.month.isin([1,2,3])) & (df.date_c.dt.year==2015)].groupby(['id','Latitude','Longitude']).mean()[['RadE9_DNB']].reset_index()
    # create pre-radiance based block column or groups
    db['block'] = db.RadE9_DNB.apply(lambda x: "Very Bright" if x>=60 else "Bright" if 40<=x<60 else "Moderate" if 20<=x<40 else "Low" if 3<=x<20 else "Very Low")
    db = db.rename(columns = {'RadE9_DNB':'rad_b'})
    print("before db prepared")
    #-------------------------------------------
    da = pd.read_pickle('post_event_instant_2')
    da = da.rename(columns = {'RadE9_DNB':'rad_a'})
    dp = pd.read_pickle('present_event_instant')
    dp = dp.rename(columns = {'RadE9_DNB':'rad_p'})
    print("loaded pickles")
    #---------calculate damage-----------------
    dd = pd.merge(db, da[['id','rad_a']], on='id')
    dd['dam_p'] = (dd['rad_a'] - dd['rad_b'])*100.0/dd['rad_b']
    dd = dd.groupby('block').mean()[['dam_p']].reset_index()
    dd['dam_p'] = dd['dam_p'].apply(lambda x: abs(x))
    print("damage calculated")
    #---------calculate recovery---------------
    dr = pd.merge(db, dp[['id','rad_p']], on='id')
    dr['rec_p'] = dr['rad_p']*100.0/dr['rad_b']
    dr = dr.groupby('block').mean()[['rec_p']].reset_index()
    print("recovery calculated")
    code.interact(local = locals())



if __name__=='__main__':
    CURRENTDIR = os.getcwd()
    re = CURRENTDIR + '/sanaa_csv/'
    dz = pd.read_excel(CURRENTDIR + '/sanaa_sat_zenith/sanaa_sat_zen.xlsx')
    #---create & pickle monthly database------
    # dg = create_daily_database(re)
    #-----read daily db pickle--------------
    dy = pd.read_pickle('yemen_daily_db')
    dy = dy.reset_index().drop(columns = 'index')
    # convert negative values to 0
    dy.RadE9_DNB = dy.RadE9_DNB.clip(lower = 0)
    # code.interact(local = locals())
    # #------create database of recovery rate--------
    # drr = recovery_rate(dy.copy(), 'daily')
    # drc = radiance_change(dy.copy())
    # da = affected_areas(dy.copy())
    # d_ur = updated_recovery(dy.copy())

    # dc = bright_centers_comparison(dy.copy())
    # histogram_plots(dy.copy(), "before_blast_radiance")
    # histogram_plots(dy.copy(), "radiance_right_after_blasts")
    # histogram_plots(dy.copy(), "radiance_today")
    # histogram_plots(dy.copy(), "radiance_change")

    # drr = pd.read_pickle('recover_rate_daily')
    # drc = pd.read_pickle('radiance_change')
    # da = pd.read_pickle('affected_areas')

    # db = impact_based_classification(dy.copy())
    # dg = track_growth_by_damage(dy.copy(),db.copy())
    # monthly_recovery_rate_by_damage(dy.copy(),db.copy())

    # di = impact_based_classification(dy.copy())
    # db = pre_event_rad_based_classification(dy.copy())

    # monthly_recovery_rate_by_rad_centers(dy.copy())
    # track_growth_by_pre_bombing(dy.copy(),db.copy())

    # db1 = post_event_rad_based_classification(dy.copy())
    # db2 = present_day_rad_based_classification(dy.copy())

    calculate_damage_recovery(dy.copy())
    code.interact(local = locals())