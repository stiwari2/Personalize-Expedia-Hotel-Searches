import math
import numpy as np
import pandas as pd
import xlearn as xl


class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row, t):
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))

        for col, val in row.loc[row.index != self.y].to_dict().items():
            col_type = t[col]
            name = '{}_{}'.format(col, val)
            if col_type.kind ==  'O':
                ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col_type.kind == 'f':
                ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        return pd.Series({idx: self.transform_row_(row, t) for idx, row in df.iterrows()})


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