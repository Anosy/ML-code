import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import csv


def load_data(file_name, is_train):
    data = pd.read_csv(file_name)
    pd.set_option('display.width', 300)
    # print(data)
    # print(data.describe())

    # 性别，进行0-1编码
    data['Sex'] = pd.Categorical(data['Sex']).codes

    # 补齐船票价格缺失，用中位数
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        for f in range(3):
            # DataFrame.dropna(axis=0, how='any', thresh=None..)  作用：判断每行是否有空值，如果有去掉该行
            # DataFrame.median()  作用：给对每行或者列取中位数
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median() # 基于Pclass，将每一行，考虑Fare列，去掉空的数后取其中位数并且赋值
        for f in range(3):
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f]  # 基于Pclass，如果有Fare有空值则将其用中位数填充
            # data[(data.Fare == 0 ) & (data.Pclass == f + 1)]['Fare'] = fare[f]

    # 年龄：使用均值来代替缺失值
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()),'Age'] = mean_age

    # 年龄：使用随机森林预测年龄的缺失值
    is_train = True
    if is_train:
        print('随机森林预测缺失年龄：--start--')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print('随机森林预测缺失年龄：--over--')
    else:
        print('随机森林预测缺失年龄2：--start--')
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print age_exist
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print('随机森林预测缺失年龄2：--over--')

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 将缺失的城市用S来填充
    # data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'U': 0}).astype(int)
    # print(data.Embarked)
    embarked_data = pd.get_dummies(data.Embarked)   # 利用get_dummies进行One-hot编码
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    # data.to_csv('New_Data.csv')  # 将数据以csv格式保存

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 通过复制来作弊
    # x = np.tile(x, (5, 1))  # 竖着复制成相同的5份
    # y = np.tile(y, (5, ))    # 横着复制成相同的5份

    if is_train:
        return x, y
    return x, data['PassengerId']


if __name__ == '__main__':
    x, y = load_data('Titanic.train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    # LogisticRegression
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_hat)

    # RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_hat, y_test)

    # xgBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'), (data_test, 'test')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)
    y_hat = bst.predict(data_test)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_hat, y_test)

    print('Logistic回归：%.3f%%' % lr_acc)
    print('随机森林：%.3f%%' % rfc_acc)
    print('XGBoost：%.3f%%' % xgb_acc)