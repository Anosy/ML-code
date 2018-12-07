import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# cabin 客舱号码
def ADSplit(s):
    """
    Function to try and extract cabin letter and number from the cabin column.
    Runs a regular expression that finds letters and numbers in the
    string. These are held in match.group, if they exist.
    """

    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)

    try:
        letter = match.group(1)
    except:
        letter = ''

    try:
        number = match.group(2)
    except:
        number = 9999

    return letter, number


def DR(s):
    """
    从cabin中提取字母，数值，以及数量
    """
    # Check contents
    if isinstance(s, (int, float)):
        # If field is empty, return nothing
        letter = ''
        number = ''
        nRooms = 9999
    else:
        # If field isn't empty, split sting on space. Some strings contain
        # multiple cabins.
        s = s.split(' ')
        # Count the cabins based on number of splits
        nRooms = len(s)
        # Just take first cabin for letter/number extraction
        s = s[0]

        letter, number = ADSplit(s)

    return [letter, number, nRooms]

def splitName(s, titleDict):
    """
    从名称中提取标题，替换为标题字典中的值。 也就是返回姓氏
    """
    # Remove '.' from name string
    s = s.replace('.', '')
    # Split on spaces
    s = s.split(' ')
    # get surname 姓
    surname = s[0]

    # 将名字中于字典匹配的进行替换，统一化
    title = [t for k, t in titleDict.items() if str(k) in s]

    # 如果不能匹配，那么就统一写成Other
    if title == []:
        title = 'Other'
    else:
        # Title is a list, so extract contents
        title = title[0]

    # Return surname (stripping remaining ',') and title as string
    return surname.strip(','), title

# 预备数据
def prepLGB(data, classCol='', IDCol='', fDrop=[]): # fDrop为要除去的列
    # Drop class column
    if classCol != '':
        labels = data[classCol]
        fDrop = fDrop + [classCol]  # 同时除去标签列
    else:
        labels = []

    if IDCol != '':
        IDs = data[IDCol]
    else:
        IDs = []

    if fDrop != []:
        data = data.drop(fDrop, axis=1)

    # Create LGB mats
    lData = lgb.Dataset(data, label=labels, free_raw_data=False,
                        feature_name=list(data.columns),
                        categorical_feature='auto')

    return lData, labels, IDs, data  # 返回LGB形式的数据，标签，ID，除去某些列的数据


def feature(trainRaw, testRaw, nTrain, full):
    ##  cabin特征
    out = full['Cabin'].apply(DR)  # 从cabin中提取字母，数值，以及数量
    out = out.apply(pd.Series)
    out.columns = ['CL', 'CN', 'nC']
    full = pd.concat([full, out], axis=1)

    ##  family 特征
    # SibSp泰坦尼克号上的兄弟姐妹/配偶；Parch泰坦尼克号上的父母/孩子的数量
    full['fSize'] = full['SibSp'] + full['Parch'] + 1
    full['fRatio'] = (full['Parch'] + 1) / (full['SibSp'] + 1)
    full['Adult'] = full['Age'] > 18
    # 从名称中提取标题，进行标准化
    titleDict = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Sir",
        "Don": "Sir",
        "Sir": "Sir",
        "Dr": "Dr",
        "Rev": "Rev",
        "theCountess": "Lady",
        "Dona": "Lady",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Lady"
    }
    out = full['Name'].apply(splitName, args=[titleDict])  # apply使用的函数，如果还有其他参数，就使用args
    out = out.apply(pd.Series)
    out.columns = ['Surname', 'Title']
    full = pd.concat([full, out], axis=1)

    ## 要重新编码的分类列表的列表
    catCols = ['Sex', 'Embarked', 'CL', 'CN', 'Surname', 'Title']
    # 将对应的类别列的编码方式进行转化，转为dtypes=category
    for c in catCols:
        # 将列转换为pd.Categorical
        full[c] = pd.Categorical(full[c])
        # 提取cat.codes并用这些替换列
        full[c] = full[c].cat.codes
        # 将类别代码转换为分类...
        full[c] = pd.Categorical(full[c])
    # 生成分类列的逻辑索引，以便稍后使用LightGBM
    catCols = [i for i, v in enumerate(full.dtypes) if str(v) == 'category']

    # 年龄特征
    # 使用中位数来补齐缺失的年龄
    full.loc[full.Age.isnull(), 'Age'] = np.median(full['Age'].loc[full.Age.notnull()])

    # 训练集和测试集再次分开
    train = full.iloc[0:nTrain, :]
    test = full.iloc[nTrain::, :]

    return train, test

if __name__ == '__main__':
    trainRaw = pd.read_csv(r'Taitanic_data/train.csv')
    testRaw = pd.read_csv(r'Taitanic_data/test.csv')
    nTrain = trainRaw.shape[0]   # 训练集的个数
    full = pd.concat([trainRaw, testRaw], axis=0)  # 合并训练集和测试集

    train, test = feature(trainRaw, testRaw, nTrain, full) # 进行特征工程


    fDrop = ['Ticket', 'Cabin', 'Name']  # 除去这三列
    trainData, validData = train_test_split(train, test_size=0.3, stratify=train.Survived)  # 使用按照label分层方式来分割
    trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,  # 训练集
                                                           classCol='Survived',
                                                           IDCol='PassengerId',
                                                           fDrop=fDrop)
    validDataL, validLabels, validIDs, validData = prepLGB(validData,  # 验证集
                                                           classCol='Survived',
                                                           IDCol='PassengerId',
                                                           fDrop=fDrop)
    testDataL, _, _, testData = prepLGB(test,  # 测试集
                                        classCol='Survived',
                                        IDCol='PassengerId',
                                        fDrop=fDrop)
    allTrainDataL, allTrainLabels, _, allTrainData = prepLGB(train,  # 全部的训练集
                                                             classCol='Survived',
                                                             IDCol='PassengerId',
                                                             fDrop=fDrop)

    # 设置参数
    params = {
        'boosting_type': 'gbdt',
        'max_depth': -1,
        'objective': 'binary',
        'nthread': 4,  # Updated from nthread
        'num_leaves': 64,
        'learning_rate': 0.05,
        'max_bin': 512,
        'subsample_for_bin': 200,
        'subsample': 1,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 5,
        'reg_lambda': 10,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 5,
        'scale_pos_weight': 1,
        'num_class': 1,
        'metric': 'binary_error'
    }
    # grid参数矩阵
    gridParams = {
        'learning_rate': [0.005],
        'n_estimators': [8, 16, 24],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'random_state': [501],  # Updated from 'seed'
        'colsample_bytree': [0.64, 0.65, 0.66],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 1.2],
        'reg_lambda': [1, 1.2, 1.4],
    }

    mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             n_jobs=4,  # Updated from 'nthread'
                             silent=True,
                             max_depth=params['max_depth'],
                             max_bin=params['max_bin'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])
    # 获取默认参数
    mdl.get_params().keys()

    # 网格搜索
    # grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
    # grid.fit(allTrainData, allTrainLabels)
    #
    # params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    # params['learning_rate'] = grid.best_params_['learning_rate']
    # # params['max_bin'] = grid.best_params_['max_bin']
    # params['num_leaves'] = grid.best_params_['num_leaves']
    # params['reg_alpha'] = grid.best_params_['reg_alpha']
    # params['reg_lambda'] = grid.best_params_['reg_lambda']
    # params['subsample'] = grid.best_params_['subsample']

    # 网格搜索的结果
    params_grid = {'boosting_type': 'gbdt', 'max_depth': -1, 'objective': 'binary', 'nthread': 4, 'num_leaves': 8,
     'learning_rate': 0.005, 'max_bin': 512, 'subsample_for_bin': 200, 'subsample': 0.7, 'subsample_freq': 1,
     'colsample_bytree': 0.64, 'reg_alpha': 1, 'reg_lambda': 1.4, 'min_split_gain': 0.5, 'min_child_weight': 1,
     'min_child_samples': 5, 'scale_pos_weight': 1, 'num_class': 1, 'metric': 'binary_error'}

    #  包含不同训练/验证拆分的早期停止套件k模型
    k = 12
    predsValid = 0
    predsTrain = 0
    predsTest = 0
    for i in range(0, k):
        print('Fitting model', k)

        # Prepare the data set for fold
        trainData, validData = train_test_split(train, test_size=0.3, stratify=train.Survived)
        trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
                                                               classCol='Survived',
                                                               IDCol='PassengerId',
                                                               fDrop=fDrop)
        validDataL, validLabels, validIDs, validData = prepLGB(validData,
                                                               classCol='Survived',
                                                               IDCol='PassengerId',
                                                               fDrop=fDrop)
        gbm = lgb.train(params=params_grid,
                        train_set=trainDataL,
                        num_boost_round=100000,
                        valid_sets=[trainDataL, validDataL],
                        early_stopping_rounds=50,
                        verbose_eval=4 # 每个4个输出boosting阶段
                        )

        # Predict
        predsValid += gbm.predict(validData, num_iteration=gbm.best_iteration) / k
        predsTrain += gbm.predict(trainData, num_iteration=gbm.best_iteration) / k
        predsTest += gbm.predict(testData, num_iteration=gbm.best_iteration) / k
    sub = pd.DataFrame()
    sub['PassengerId'] = test['PassengerId']
    sub['Survived'] = np.int32(predsTest > 0.5)
    sub.to_csv('sub2.csv', index=False)