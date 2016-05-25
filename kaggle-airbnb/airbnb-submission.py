import pandas as pd
import numpy as np
import zipfile
from sklearn.preprocessing import LabelEncoder

local_path = '/Users/eloiseheydenrych/Downloads'

# get train data
z = zipfile.ZipFile(local_path + '/train_users_2.csv.zip')
df = pd.read_csv(z.open('train_users_2.csv'), parse_dates=[1,2])

# Our Y values
ys = df_train['country_destination'].values

# get test data
z = zipfile.ZipFile(local_path + '/test_users.csv.zip')
df_testusers = pd.read_csv(z.open('test_users.csv'),parse_dates=[1,2])

# get sessions data
z = zipfile.ZipFile(local_path + '/sessions.csv.zip')
df_sessions = pd.read_csv(z.open('sessions.csv'))
