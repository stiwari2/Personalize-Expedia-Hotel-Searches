import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from datetime import datetime
import random
import warnings
import xlearn as xl
import math
from sklearn.metrics import ndcg_score
from kmodes.kprototypes import KPrototypes

random.seed(111)
PATH = "training_set_.csv"
PATH2 = "test_set.csv"

sns.set(style="ticks", color_codes=True)

np.set_printoptions(threshold=sys.maxsize)  # show full table
np.seterr(divide='ignore', invalid='ignore')  # no warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('float_format', '{:f}'.format)
pd.set_option('display.width', 10000)
pd.options.display.max_colwidth = 10000

dftrain = pd.read_csv('dataprep.csv', index_col=0)
print(dftrain.min())
print(dftrain.max())
print(dftrain.head())
print(dftrain[pd.isnull(dftrain).any(axis=1)])

def groupsize(df):
    srch_value = df.srch_id.value_counts()
    df_srch_count = pd.DataFrame([srch_value]).T.sort_index()

    return df_srch_count.srch_id


params = {'objective': 'rank:ndcg'}
xgb_rank = xgb.XGBRanker(**params)
x_train = dftrain.drop(['click_bool', 'booking_bool'], axis=1)
y_train = np.array(dftrain['click_bool']) + 4*np.array(dftrain['booking_bool'])
x_test = x_train[4500022:4958347].copy()
y_test = y_train[4500022:4958347].copy()
x_val = x_train[4000003:4500022].copy()
y_val = y_train[4000003:4500022].copy()
x_train = x_train[0:4000003]
y_train = y_train[0:4000003]

resultTest = x_test[['srch_id', 'prop_id']].copy()
resultTest['click_bool'] = np.array(dftrain.loc[4500022:4958347, 'click_bool']) + 4*np.array(dftrain.loc[4500022:4958347, 'booking_bool'])

xgb_rank.fit(x_train,y_train,groupsize(x_train), groupsize(x_train), eval_set=[(x_val, y_val)],
              eval_group=[groupsize(x_val)], eval_metric='ndcg@5')
evals_result = xgb_rank.evals_result
print(evals_result['eval_0']['ndcg@5'][-1])

preds = xgb_rank.predict(x_test)[0:458325]

print("Created and trained model")

resultTest["prediction"] = np.array(preds)

resultTest = resultTest.set_index('srch_id')


def dcg_equation(data):
    length = data.shape[0]
    realSortedData = data.sort_values(by=['click_bool'], ascending=False)
    realResult = 0
    predResult = 0
    for index in range(0, min(length, 5)):
        predResult += data.iloc[index][['click_bool']] / math.log2(index + 2)
        realResult += realSortedData.iloc[index][['click_bool']] / math.log2(index + 2)
    return predResult/realResult


uniqueTestIds = resultTest.index.unique()
accuracy = np.zeros(len(uniqueTestIds))
for i in range(len(uniqueTestIds)):
    customerData = resultTest.loc[uniqueTestIds[i]]  # only data from 1 search id
    customerData = customerData.sort_values(by=['prediction'], ascending=False)  # sort on prediction
    accuracy[i] = dcg_equation(customerData)

print(accuracy.mean())


import math
import numpy as np
import pandas as pd

ffm_train = FFMFormatPandas()
ffm_train_data = ffm_train.fit_transform(X_train.reset_index().drop(['srch_id'], axis= 1), y='click_bool')
ffm_test_data = ffm_train_data[200009:251816]
ffm_train_data = ffm_train_data[0:200009]

with open(r'train_ffm.txt', 'w') as f:
    f.write(ffm_train_data.to_string(header = False, index = False))
with open(r'test_ffm.txt', 'w') as f:
    f.write(ffm_test_data.to_string(header = False, index = False))

ffm_model = xl.create_ffm()

ffm_model.setTrain("train_ffm.txt")
ffm_model.setValidate("test_ffm.txt")
ffm_model.disableLockFree()

param = {'task': 'binary',
         'lr': 0.1,
         'lambda': 0.2,
         'metric': 'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, './model.out')

# Prediction task
ffm_model.setTest("test_ffm.txt")  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./model.out", "./output.txt")
resultTest = X_test[['prop_id', 'click_bool']]
prediction = pd.read_csv("./output.txt", header=None)
resultTest["prediction"] = np.array(prediction)


ffm_train2 = FFMFormatPandas()
ffm_train2_data = ffm_train.fit_transform(X_train.reset_index().drop(['srch_id'], axis= 1), y='booking_bool')
ffm_test2_data = ffm_train2_data[200009:251816]
ffm_train2_data = ffm_train2_data[0:200009]
with open(r'train_ffm2.txt', 'w') as f:
    f.write(ffm_train2_data.to_string(header = False, index = False))
with open(r'test_ffm2.txt', 'w') as f:
    f.write(ffm_test2_data.to_string(header = False, index = False))

ffm_model2 = xl.create_ffm()

ffm_model2.setTrain("train_ffm2.txt")
ffm_model2.setValidate("test_ffm2.txt")
ffm_model2.disableLockFree()

param = {'task': 'binary',
         'lr': 0.1,
         'lambda': 0.002,
         'metric': 'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model2.fit(param, './model2.out')

# Prediction task
ffm_model2.setTest("test_ffm2.txt")  # Test data
ffm_model2.setSigmoid()  # Convert output to 0-1


def dcg_equation(data):
    relevance = data.shape[0]
    realSortedData = data.sort_values(by=['click_bool'], ascending=False)
    realResult = 0
    predResult = 0
    for index in range(0, min(relevance, 5)):
        predResult += data.iloc[index][['click_bool']] / math.log2(index + 2)
        realResult += realSortedData.iloc[index][['click_bool']] / math.log2(index + 2)
    return predResult/realResult


# Start to predict
# The output result will be stored in output.txt
ffm_model2.predict("./model2.out", "./output2.txt")
prediction2 = pd.read_csv("./output2.txt", header=None)
resultTest["prediction"] = 4*np.array(prediction)
resultTest['click_bool'] = resultTest['click_bool'] + 4*X_test['booking_bool']
uniqueTestIds = resultTest.index.unique()
accuracy = np.zeros(len(uniqueTestIds))
for i in range(len(uniqueTestIds)):
    customerData = resultTest.loc[uniqueTestIds[i]]
    print(customerData)
    customerData = customerData.sort_values(by=['prediction'], ascending=False)
    accuracy[i] = dcg_equation(customerData)

print(accuracy.mean())