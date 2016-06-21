# Thank you to Christian Hirsch for inspiration for some of the features used in this code.

import pandas as pd
import numpy as np
import math
from xgboost.sklearn import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('/Users/eloiserosen/Downloads/train.csv')
df_test = pd.read_csv('/Users/eloiserosen/Downloads/test.csv')


#####################################
# INITIAL DATA CLEANING AND FEATURES
#####################################

def recenter_cyclic(data, period):
    return (data - period/2) % period

def clean_data(df):
    feature_list=df.columns.tolist()
    
    # drop columns we don't need
    if 'Descript' in feature_list:
        del df['Descript']
    if 'Resolution' in feature_list:
        del df['Resolution']
    
    # create columns based on timestamp
    date_time = pd.to_datetime(df['Dates'])
    year = date_time.dt.year
    df['Year'] = year
    month = date_time.dt.month
    df['Month'] = month
    week = date_time.dt.week
    df['Week'] = week
    day = date_time.dt.day
    df['Day'] = day
    hour = date_time.dt.hour
    df['Hour'] = hour
    #some crimes are logged at a precise time. Others, like some thefts, have rounded time
    minute = date_time.dt.minute - 30
    df['Minute'] = minute
    #time = hour*60+minute # counting minutes
    #df['Time'] = time
    
    df['Minute_Cyclic'] = recenter_cyclic(df['Minute'], 60)
    df['Hour_Cyclic'] = recenter_cyclic(df['Hour'], 24)
    df['Week_Cyclic'] = recenter_cyclic(df['Week'], 52)
    df['Month_Cyclic'] = recenter_cyclic(df['Month'], 12)
    
    # column to indicate if address was on a block
    df['StreetCorner'] = df['Address'].str.contains('/').map(int)
    
    return df

df =clean_data(df)
df_test = clean_data(df_test)


#######################
# GROUP CRIME FEATURE
# Some types of crimes are more likely to happen in groups than others. So, counting up the number of crimes that are
# logged for the same time and place gives us valuable information about what kind of crime likely occurred
######################
def group_crime(df):
    groupedFrame = df.reset_index().groupby(['Dates', 'X', 'Y'])['index'] 
    return groupedFrame.transform(lambda x: len(list(x)))

reshaper = FunctionTransformer(lambda X: X.reshape(-1,1),validate=False)
group_crime_function = FunctionTransformer(lambda df : group_crime(df), validate = False)
group_crime_pipeline = Pipeline([('group_crime_function', group_crime_function), ('reshaper', reshaper)])
group_crime_names = ['CollocatedCrime'] 

df_new = group_crime(df)
df = df.join(df_new)
df.rename(columns={'index': 'group_crime'}, inplace=True)

df_test_new = group_crime(df_test)
df_test = df_test.join(df_test_new)
df_test.rename(columns={'index': 'group_crime'}, inplace=True)



####################################
# COUNT FEATURIZATION OF ADDRESSES
####################################

# idea from kaggle user papadopc
# process: count categories for each address, get log of the relative frequency, get log-ratios
# replace log of 0 with large negative number
# to ensure that most entries are 0, subtract that number from pivot table afterwards

#minimum number of crimes that are required for log-ratios to be computed
MIN_COUNT = 5

#default log ratio
rare_cats = set(['FAMILY OFFENSES', 'BAD CHECKS', 'BRIBERY', 'EXTORTION',
       'SEX OFFENSES NON FORCIBLE', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT',
       'TREA'])
all_cats = set(df['Category'].unique())
common_cats = all_cats-rare_cats
DEFAULT_RAT = math.log(1.0 / len(common_cats))

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[data_dict.columns.intersection(self.key)]
    
    
class CountFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_ratios = None

    def fit(self, data, y = None):
        
        #determine total number of crimes
        address_counts = pd.DataFrame(data.groupby(['Address']).size(), columns = ['total_count'])
        
        #determine number of crimes by category and address
        address_category_counts = pd.DataFrame(data.groupby(['Address', 'Category']).size(), columns = ['count']).reset_index()
        address_category_counts['total_count'] = address_category_counts.groupby('Address')['count'].transform(lambda x: max(MIN_COUNT, x.sum()))
        address_category_counts = address_category_counts[address_category_counts['count'] >= MIN_COUNT]
        address_category_counts['log'] = (address_category_counts['count']/address_category_counts['total_count']).apply(math.log)
        
        #pivot table
        self.log_ratios = pd.pivot_table(address_category_counts, index = 'Address', 
                          columns = 'Category', values = 'log', fill_value = DEFAULT_RAT) - DEFAULT_RAT  
        self.log_ratios.columns = ['LogRatio_' + str(x) for x in range(len(self.log_ratios.columns))]
        
        #join total counts
        self.log_ratios = self.log_ratios.merge(address_counts, 'left', left_index = True, right_index = True)
        return self

    def transform(self, data):
        
        #merge with log_ratios
        merged_data = data.loc[:, 'Address'].reset_index().merge(self.log_ratios, 'left', left_on = 'Address', right_index = True).iloc[:,2:]        
        
        #replace NAs with default values
        default_df = pd.DataFrame(np.zeros(merged_data.shape), columns = merged_data.columns) 
        default_df.iloc[:, -1] = MIN_COUNT

        return merged_data.combine_first(default_df)


address_cat_is = ItemSelector(['Address', 'Category'])
count_feat_fun = CountFeatureEncoder()

count_feat_pipe = Pipeline([('address_cat_is', address_cat_is), 
                            ('count_feat_fun', count_feat_fun)])
                              
count_feat_names = ['count_feature_' + i for i in common_cats] + ['total_count']
countfeatencoder = CountFeatureEncoder()
countfeatencoder.fit(df)
countfeatencoder_df = countfeatencoder.transform(df)

df = df.join(countfeatencoder_df)
countfeatencoder_df_test = countfeatencoder.transform(df_test)

df_test = df_test.join(countfeatencoder_df_test)
df_test.head()


df.drop(['Dates', 'Month', 'Month_Cyclic'], axis=1, inplace=True)
df_test.drop(['Dates', 'Month', 'Month_Cyclic'], axis=1, inplace=True)




####################################
# DUMMIES FOR CATEGORICAL VARIABLES
####################################

dummy_DayOfWeek = pd.get_dummies(df['DayOfWeek'], prefix='Day')
del dummy_DayOfWeek['Day_Friday']
del df['DayOfWeek']
df = df.join(dummy_DayOfWeek)
dummy_PdDistrict = pd.get_dummies(df['PdDistrict'], prefix='District')
del dummy_PdDistrict['District_SOUTHERN']
del df['PdDistrict']
df = df.join(dummy_PdDistrict)


dummy_DayOfWeek = pd.get_dummies(df_test['DayOfWeek'], prefix='Day')
del dummy_DayOfWeek['Day_Friday']
del df_test['DayOfWeek']
df_test = df_test.join(dummy_DayOfWeek)
dummy_PdDistrict = pd.get_dummies(df_test['PdDistrict'], prefix='District')
del dummy_PdDistrict['District_SOUTHERN']
del df_test['PdDistrict']
df_test = df_test.join(dummy_PdDistrict)



#############################################
# MEDIAN IMPUTATION FOR LATITUDE AND LONGITUDE
# as noted in data exploration file, there are some latitude and longitude values that are obviously incorrect. 
#Impute these with the median.
#############################################

# fill incorrect values with NaN
df['X'].replace(-120.5, np.nan, inplace = True)
df['Y'].replace(90, np.nan, inplace = True)

# find median for median imputation. Save values so I can reuse for test file.
medianX = df['X'].median()
medianY = df['Y'].median()

# median imputation
df['X'] = df['X'].fillna(medianX)
df['Y'] = df['Y'].fillna(medianY)

#median imputation in test file
# fill incorrect values with NaN
df_test['X'].replace(-120.5, np.nan, inplace = True)
df_test['Y'].replace(90, np.nan, inplace = True)

# median imputation
df_test['X'] = df_test['X'].fillna(medianX)
df_test['Y'] = df_test['Y'].fillna(medianY)



##################################
# TARGET VECTOR AND FEATURE MATRIX
##################################

#target vector y
y = df['Category']
y.head()

#Matrix of X's.
X = df
del X['Category']
X.head()

# we no longer need these cols
del df['Address']
del df_test['Address']


#############################################################
# SCALE CONTINUOUS FEATURES WITH ZERO MEAN AND UNIT VARIANCE
#############################################################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True)  
# make a df called x_continous that has just our continous features
ContinuousCols = ['X','Y', 'Year', 'Week', 'Day', 'Hour', 'Minute', 'Minute_Cyclic', 'Hour_Cyclic', 'Week_Cyclic', 'group_crime', 'total_count']
X_continuous = X[ContinuousCols]

# scale to zero mean and unit variance
X_continuous = scaler.fit(X_continuous).transform(X_continuous)
X_continuous = pd.DataFrame(X_continuous, columns = ContinuousCols)

# delete unscaled cols form original X df
X = X.drop(ContinuousCols, axis=1)

# merge 
X = pd.concat([X_continuous, X], axis=1)
X.head()

# scale test data with zero mean and unit variance as well. Use same scaler object I created on my training data.
kaggle_X = df_test

# make a df called x_continous that has just our continous features
kaggle_X_continuous = kaggle_X[ContinuousCols]
# scale to zero mean and unit variance
kaggle_X_continuous = scaler.transform(kaggle_X_continuous)
kaggle_X_continuous = pd.DataFrame(kaggle_X_continuous, columns = ContinuousCols)
# delete unscaled cols form original kaggle_X df
kaggle_X = kaggle_X.drop(ContinuousCols, axis=1)

# merge 
kaggle_X = pd.concat([kaggle_X_continuous, kaggle_X], axis=1)
kaggle_X.head()


##############################
# CLASSIFIER AND PREDICTIONS
##############################

# delete the id column for now so that we can run our classifier
ids = kaggle_X['Id']
del kaggle_X['Id']

xgb = XGBClassifier(objective = 'multi:softprob', max_depth = 6, learning_rate = 1.0, max_delta_step = 1, seed=0)
xgb.fit(X, y)
predictions = pd.DataFrame(xgb.predict_proba(kaggle_X), columns=xgb.classes_)

# grid search below
'''
xgb = XGBClassifier()

from sklearn.grid_search import GridSearchCV
param_grid = {'max_depth': np.arange(3, 12)}
grid = GridSearchCV(xgb, param_grid, n_jobs=4)
grid.fit(X, y)
print grid.grid_scores_
print grid.best_score_
print grid.best_estimator_
print grid.best_params_
'''

predictions.head()

# put the id column back
predictions = pd.concat([ids, predictions], axis=1)
predictions.head()

# create submission file
predictions.to_csv('submission.csv',index=False)