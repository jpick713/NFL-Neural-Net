# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 20:29:47 2019

@author: jpick
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.compose import ColumnTransformer
import pdpipe as pdp
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import mean_squared_error
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import kerastuner as kt
from collections import Counter
from scipy.sparse import csr_matrix


np.random.seed(123)
# download data and look at shape and types
data = pd.read_csv('csv files for training/NFL_train/train.csv')
print(data.shape)
data.dtypes

# convert clock string to a time in mins
data['GameClock'][15:35]
def time_convert(time):
    new_time=time.split(":")
    time_in_mins=int(new_time[0])
    time_in_secs=int(new_time[1])/60
    return time_in_mins + time_in_secs

trial_time=data['GameClock'].apply(time_convert)
data['GameClock']=trial_time
data.dtypes
data['StadiumType'].unique()
c=Counter(data['StadiumType'])
print(c)
# Drop redundant columns or those which shoudl not contribute to the yardage
y=data['Yards']
X=pdp.ColDrop(['PlayerBirthDate', 'PlayerCollegeName', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium',
               'Location', 'StadiumType', 'WindDirection','TimeHandoff', 'TimeSnap', 'DisplayName', 
               'JerseyNumber', 'GameId', 'PlayId', 'Yards', 'WindSpeed', 'Week', 'GameWeather', 'Humidity']).apply(data)

# look at object datatypes and start cleaning process
def weather_change(x,y):
    if y in ['Indoor', 'Indoors', 'Dome', 'Domed, closed', 'Retr. Roof-Closed', 'Retr. Roof - Closed',
             'Closed Dome', 'Dome, closed', 'Domed', 'Indoor, Roof Closed', 'Retr. Roof Closed','Bowl']:
        return 'Clear'
    else:
        return x
 
data['GameWeather']=data.apply(lambda x: weather_change(x['GameWeather'],x['StadiumType']), axis=1)
data['GameWeather'].unique()
weather_check=data[data['StadiumType']=='Indoor'].GameWeather
weather_check.unique()

# Check composition of these columns and convert info to one overall YardLine col
X['PossessionTeam'].unique()
X['FieldPosition'].unique()
X['YardLine'].unique()

def overall_yard_line(x,y,z):
    if y == z or x==50:
        return x
    else:
        return x+50
    
X['OverallYardLine']=X.apply(lambda x: overall_yard_line(x['YardLine'], x['FieldPosition'], x['PossessionTeam']), axis=1)
X['OverallYardLine'].unique()
X=pdp.ColDrop(['PossessionTeam', 'FieldPosition', 'YardLine']).apply(X)

#look at other object columns
c=Counter(X['OffenseFormation'])
print(c)
c=Counter(X['OffensePersonnel'])
print(c)
c=Counter(X['DefensePersonnel'])
print(c)
c=Counter(X['PlayDirection'])
print(c)
c=Counter(X['PlayerHeight'])
print(c)

# convert height to inches
def height_convert(height):
    feet= int(height.split('-')[0])
    inches= int(height.split('-')[1])
    return 12*feet + inches

new_height= X['PlayerHeight'].apply(height_convert)
X['PlayerHeight']= new_height


# aggregate rare defenses into Other category
def defense_personnel(x):
    if x in ['5 DL, 1 LB, 5 DB', '6 DL, 2 LB, 3 DB', '1 DL, 5 LB, 5 DB', '2 DL, 5 LB, 4 DB', '1 DL, 3 LB, 7 DB',
             '2 DL, 2 LB, 7 DB', '3 DL, 1 LB, 7 DB', '5 DL, 5 LB, 1 DB', '5 DL, 3 LB, 2 DB, 1 OL', '0 DL, 5 LB, 6 DB',
             '4 DL, 5 LB, 2 DB', '0 DL, 4 LB, 7 DB', '2 DL, 4 LB, 4 DB, 1 OL', '5 DL, 4 LB, 1 DB, 1 OL', '4 DL, 0 LB, 7 DB',
             '4 DL, 6 LB, 1 DB', '0 DL, 6 LB, 5 DB', '4 DL, 5 LB, 1 DB, 1 OL', '6 DL, 1 LB, 4 DB', '3 DL, 4 LB, 3 DB, 1 OL',
             '1 DL, 2 LB, 8 DB', '7 DL, 2 LB, 2 DB']:
        return 'Other'
    else:
        return x
    
X['DefensePersonnel']=X.apply(lambda x: defense_personnel(x['DefensePersonnel']), axis=1)

# aggregate rare offenses into Other category
def offense_personnel(x):
    if x in ['1 RB, 1 TE, 2 WR,1 DL', '7 OL, 1 RB, 0 TE, 2 WR', '3 RB, 1 TE, 1 WR', '2 QB, 2 RB, 1 TE, 1 WR', '1 RB, 3 TE, 0 WR,1 DL',
             '6 OL, 1 RB, 2 TE, 0 WR,1 DL', '3 RB, 0 TE, 2 WR', '2 QB, 1 RB, 2 TE, 1 WR', '7 OL, 1 RB, 2 TE, 0 WR', '6 OL, 1 RB, 2 TE, 0 WR,1 LB',
             '7 OL, 2 RB, 0 TE, 1 WR', '1 RB, 1 TE, 2 WR,1 DB', '1 RB, 2 TE, 1 WR,1 LB', '6 OL, 2 RB, 1 TE, 0 WR,1 DL', '6 OL, 1 RB, 1 TE, 1 WR,1 DL',
             '1 RB, 4 TE, 0 WR', '2 QB, 1 RB, 0 TE, 3 WR', '1 RB, 3 TE, 0 WR,1 LB', '0 RB, 2 TE, 3 WR', '3 RB, 2 TE, 0 WR', '0 RB, 3 TE, 2 WR',
             '1 RB, 0 TE, 3 WR,1 DB', '1 RB, 2 TE, 1 WR,1 DB', '7 OL, 2 RB, 1 TE, 0 WR', '0 RB, 0 TE, 5 WR', '1 RB, 1 TE, 2 WR,1 LB',
             '2 RB, 2 TE, 0 WR,1 DL', '2 QB, 2 RB, 0 TE, 2 WR', '2 QB, 1 RB, 3 TE, 0 WR', '2 RB, 3 TE, 1 WR', '1 RB, 2 TE, 3 WR',
             '1 RB, 3 TE, 0 WR,1 DB', '6 OL, 0 RB, 2 TE, 2 WR', '2 QB, 2 RB, 2 TE, 0 WR', '2 QB, 3 RB, 1 TE, 0 WR', '2 RB, 1 TE, 1 WR,1 DB',
             '6 OL, 3 RB, 0 TE, 1 WR', '6 OL, 1 RB, 1 TE, 0 WR,2 DL']:
        return 'Other'
    else:
        return x

X['OffensePersonnel']=X.apply(lambda x: offense_personnel(x['OffensePersonnel']), axis=1)

# remove few NaNs from OffenseFormation with mode
def remove_formation_nan(x):
    if pd.isnull(x):
        return 'SINGLEBACK'
    else:
        return x
X['OffenseFormation']=X.apply(lambda x: remove_formation_nan(x['OffenseFormation']), axis=1)

# cleanup remaining numeric columns with mean SimpleImputer Strategy
imputer=SimpleImputer() 
null_cols=[col for col in X.columns if X[col].isnull().any()]
print(null_cols)
X_numeric=X[null_cols]

X_numeric=pd.DataFrame(imputer.fit_transform(X_numeric), columns=null_cols)
for col in null_cols:
    X[col]=X_numeric[col]

# normalize the numeric data and encode the categorical data
numeric_columns=X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns=[col for col in X.columns if col not in numeric_columns]
non_numeric=['HomeScoreBeforePlay','VisitorScoreBeforePlay']
categorical_columns=categorical_columns[:-1]
categorical_columns.append('HomeScoreBeforePlay')
categorical_columns.append('VisitorScoreBeforePlay')
ordinal_columns=['Quarter', 'Down']
standardize_columns=[col for col in numeric_columns if (not (col in non_numeric or col in ordinal_columns))]

preprocessor= ColumnTransformer(transformers=[('num',StandardScaler(), standardize_columns),
                                              ('ord', OrdinalEncoder(), ordinal_columns),
                                              ('cat', OneHotEncoder(drop='first'), categorical_columns)])
scaler_encoder=preprocessor.fit(X)

X_scl_enc=preprocessor.fit_transform(X)

y_shift=y+99
y_enc=tf.keras.utils.to_categorical(y_shift, num_classes=199)

# make a vector to select train and test sets
chooser=np.random.choice([0,1], size=X.shape[0], p=[0.1,0.9])
X_train=X_scl_enc[chooser==1]
X_val=X_scl_enc[chooser==0]
y_train=y_enc[chooser==1]
y_val=y_enc[chooser==0]
yards=X['OverallYardLine'].values
yards_train=yards[chooser==1]
yards_val=yards[chooser==0]
X_train.shape[1]
# start building the model and custom loss functions 

  ''' custom loss to train model according to how judging will 
        be performed. I have to make a CDF and I also zero out impossible yard
        predictions and re-scale to a new pmf which I cumsum to a CDF
    '''
    
def custom_loss(y_pred,y_actual, yardline):
    index=np.argmax(y_actual)
    for pred_yards in range(199):
        if ( pred_yards+ yardline <0) or (pred_yards+yardline>198):
            y_pred[pred_yards]=0
        if pred_yards>index:
            y_actual=1
    new_total_prob= K.sum(y_pred)
    y_pred= y_pred/new_total_prob
    cdf_pred= np.cumsum(y_pred)
    return K.mean(mean_squared_error(y_actual, cdf_pred))


def build_model(hp):
    input_layer= Input((179,))
    y_actual= Input(199,)
    yards= Input(1,)
    for i in range(hp.Int('num_layers', 2,6)):
        if i==0:
            x=Dense(units=hp.Int('units_0',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu')(input_layer)
        else:
            x= Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu')(x)
            if (i==3 or i==5):
               x=Dropout(hp.Choice('dropout_'+ str(i),[0.15,0.25]))(x)
    output_layer = Dense(199, activation='softmax')(x)
    model=Model(inputs=[input_layer, y_actual, yards], outputs=[output_layer])
    loss= custom_loss(output_layer, y_actual, yards)
    model.add_loss(loss)
    model.add_metrics()
    model.compile(optimizer='adam')
    
    # keras tuner object
 tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_loss',
        max_epochs=5,
        factor=2,
        hyperband_iterations=3,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        directory='.',
        project_name='NFL')
 
 tuner.search(X_train, y_train, yards_train,
             epochs=5,
             validation_data=(X_val, y_val, yards_val))
    
    
    

