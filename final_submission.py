import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# dummy
def get_values_dict(string, var):
    values_dict = {}
    for item in string.split(','):
        values_dict[var+'_'+item] = 1
    return values_dict

def get_dummy_df(df, var):
    values_list = []
    for row in df[var].apply(get_values_dict, args=(var,)):
        values_list.append(row)
    return pd.DataFrame(values_list)

	
# attributes generation in training data
df_genres_dy = get_dummy_df(df_train, 'genres')
df_tags_dy = get_dummy_df(df_train, 'tags')
df_categories_dy = get_dummy_df(df_train, 'categories')

df_train_dy = pd.concat([df_train.drop(['genres','tags','categories'],axis=1),df_genres_dy], axis=1)
df_train_dy = pd.concat([df_train_dy, df_tags_dy], axis=1)
df_train_dy = pd.concat([df_train_dy, df_categories_dy], axis=1)

df_train_dy['days_after_release'] = pd.to_datetime(df_train_dy['purchase_date']) - pd.to_datetime(df_train_dy['release_date'])
df_train_dy['days_after_release'] = df_train_dy['days_after_release'].apply(lambda x: x.days)

df_train_dy['total_positive_reviews'].fillna(0, inplace=True)
df_train_dy['total_negative_reviews'].fillna(0, inplace=True)

df_train_dy['pos_ratio'] = df_train_dy['total_positive_reviews'] / (df_train_dy['total_positive_reviews']+df_train_dy['total_negative_reviews'])

# attributes feneration in test data
df_test_genres_dy = get_dummy_df(df_test, 'genres')
df_test_tags_dy = get_dummy_df(df_test, 'tags')
df_test_categories_dy = get_dummy_df(df_test, 'categories')

df_test_dy = pd.concat([df_test.drop(['genres','tags','categories'],axis=1),df_test_genres_dy], axis=1)
df_test_dy = pd.concat([df_test_dy, df_test_tags_dy], axis=1)
df_test_dy = pd.concat([df_test_dy, df_test_categories_dy], axis=1)

df_test_dy['total_positive_reviews'].fillna(0, inplace=True)
df_test_dy['total_negative_reviews'].fillna(0, inplace=True)

df_test_dy['pos_ratio'] = df_test_dy['total_positive_reviews'] / (df_test_dy['total_positive_reviews']+df_test_dy['total_negative_reviews'])

df_test_dy['days_after_release'] = pd.to_datetime(df_test_dy['purchase_date']) - pd.to_datetime(df_test_dy['release_date'])
df_test_dy['days_after_release'] = df_test_dy['days_after_release'].apply(lambda x: x.days)

# feature selection
df_iv = pd.read_excel('df_iv.xlsx')
iv_features = df_iv[df_iv['iv']>0.02]['var_name'].values
features = [i for i in iv_features if i in df_test_dy.columns]

# data processing
data_train = df_train_dy.drop(['id','purchase_date','release_date'], axis=1)
#data_train.fillna(0, inplace=True)
data_train['is_play'] = np.where(data_train['playtime_forever']>0,1,0)
data_train_play = data_train[data_train['playtime_forever']>0]

# modeling
clf = lgb.LGBMClassifier(bagging_fraction=0.7,
                                   bagging_freq=6,
                                   cat_smooth=10,
                                   feature_fraction=0.6,
                                   lambda_l1=0.5,
                                   lambda_l2=35,
                                   learning_rate=0.05,
                                   n_estimators=600,
                                   max_depth=2,
                                   num_leaves=16,
                                   metric = 'auc')
model = lgb.LGBMRegressor(bagging_fraction=0.8,
                                   bagging_freq=6,
                                   cat_smooth=15,
                                   feature_fraction=0.95,
                                   lambda_l1=0,
                                   lambda_l2=35,
                                   learning_rate=0.05,
                                   n_estimators=100,
                                   max_depth=2,
                                   num_leaves=16)
clf.fit(data_train[features], data_train['is_play'])
model.fit(data_train_play[features], data_train_play['playtime_forever'])


df_test_dy['playtime_forever'] = model.predict(df_test_dy[features]) * clf.predict_proba(df_test_dy[features])[:,1] 
df_test_dy[['id','playtime_forever']].to_csv('submission_1129.csv',index=False)