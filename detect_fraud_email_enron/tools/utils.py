#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:55:11 2018

@author: gotamist
"""

def clean_and_move_up_poi(df):
    ''' Make 'poi' the first feature, remove lines having just nans
    '''
    import pandas as pd
    import numpy as np

    f_list = list(df)
    x_list = f_list #I need the x_list later.  Not just to move up the poi
    poi_series = df[ 'poi' ]
    poi_series = poi_series.astype('int')
    poi_df = poi_series.to_frame()
    x_list.remove('poi')
    x_list.remove('email_address')
    f_list = [ 'poi' ]+x_list
    df = df.loc[:, x_list]
    df=df.replace('NaN', np.nan)
    df=df.dropna( how ='all')
    df = poi_df.join( df, how = 'right')    #if not dropping NaN here, right or left join does not matter
    return df

def scale_features(df, col_list):
    import numpy as np
    for col in col_list:
        maxim = np.max( df[ col ] )
        minim = np.min( df[ col ] )
        if maxim==minim:
            print("no variation in feature ", col), ". Drop it!"
            return df
        else:
            df[ col ]  = ( df[ col ]  - minim ) / (maxim - minim)
    return df
    