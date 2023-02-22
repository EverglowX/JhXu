import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# 数据读取与准备
font = dict(family='Times New Roman', fontsize=10)
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

path = 'D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/'
voice_data = pd.read_csv(path + 'q1_voice.csv', encoding='utf-8')
surfing_data = pd.read_csv(path + 'q1_surfing.csv', encoding='utf-8')
voice_head = voice_data.iloc[:, 4:].columns
surfing_head = surfing_data.iloc[:, 4:].columns
for index, col in voice_data.items():
    voice_data[index].fillna(0, inplace=True)
for index, col in surfing_data.items():
    surfing_data[index].fillna(0, inplace=True)
voice_data = np.array(voice_data)
surfing_data = np.array(surfing_data)

# 划分训练数据与标签
x1 = voice_data[:, 4:]
y1 = voice_data[:, 0: 4].astype(int)
x2 = surfing_data[:, 4:]
y2 = surfing_data[:, 0: 4].astype(int)
# 划分训练集与验证集(9:1)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1, y1, test_size=0.1, random_state=42)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2, y2, test_size=0.1, random_state=42)

# 标签统计
def label_summary(label):
    labels = np.zeros(10)
    for i in range(10):
        for j in range(len(label)):
            labels[i] += np.sum(label[j] == i + 1)
    return labels

# GBDT参数调整
def gbdt_optim(feature_train, target_train):
    gbdt = GradientBoostingClassifier()
    
    gbdt_param_grid1 = {'n_estimators':[50, 100, 200, 400, 800]}
    gbdt_best = GridSearchCV(gbdt,param_grid = gbdt_param_grid1, cv = 3, scoring='accuracy', n_jobs= -1, verbose=100)
    gbdt_best.fit(feature_train,target_train)
    
    gbdt_param_grid2 = {'max_depth': [5, 10, 15, 20, 25],
                        'min_samples_split': range(100, 801, 200)}
    gbdt_best2 = GridSearchCV(gbdt_best.best_estimator_, param_grid = gbdt_param_grid2, cv = 3, scoring='accuracy', n_jobs= -1, verbose=100)
    gbdt_best2.fit(feature_train, target_train)
    
    gbdt_param_grid3 = {'min_samples_split': range(300, 1701, 200),
                        'min_samples_leaf': range(60, 101, 10)}
    gbdt_best3 = GridSearchCV(gbdt_best2.best_estimator_, param_grid = gbdt_param_grid3, cv = 3, scoring='accuracy', n_jobs= -1, verbose=100)
    gbdt_best3.fit(feature_train, target_train)
    
    gbdt_param_grid4 = {'max_features': range(7, 20, 2)}
    gbdt_best4 = GridSearchCV(gbdt_best3.best_estimator_, param_grid = gbdt_param_grid4, cv = 3, scoring='accuracy', n_jobs= -1, verbose=100)
    gbdt_best4.fit(feature_train, target_train)
    
    gbdt_param_grid5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
    gbdt_best5 = GridSearchCV(gbdt_best4.best_estimator_, param_grid = gbdt_param_grid5, cv = 3, scoring='accuracy', n_jobs= -1, verbose=100)
    gbdt_best5.fit(feature_train, target_train)
    
    gbdt_param_grid6 = {'learning_rate':[0.001, 0.01, 0.1, 0.25]}
    gbdt_best6 = GridSearchCV(gbdt_best5.best_estimator_, param_grid = gbdt_param_grid6,cv = 3,scoring='accuracy',n_jobs= -1, verbose=100)
    gbdt_best6.fit(feature_train, target_train)
    
    # 输出最佳模型参数
    print(gbdt_best6.best_estimator_)
    print(gbdt_best6.best_params_, '  ','score: ', gbdt_best6.best_score_)
    
    return gbdt_best6.best_params_, gbdt_best6

# RandomForest参数调整
def rf_optim(feature_train, target_train):
    rf = RandomForestClassifier(criterion='gini')
    
    rf_param_grid1 = {'n_estimators':[100, 200, 400, 800, 1000, 1200, 1500]}
    rf_best = GridSearchCV(rf, param_grid=rf_param_grid1, cv=3, scoring='accuracy', n_jobs=-1, verbose=100)
    rf_best.fit(feature_train, target_train)
    
    rf_param_grid2 = {'min_samples_split': np.arange(5, 101, 5)}
    rf_best2 = GridSearchCV(rf_best.best_estimator_, param_grid=rf_param_grid2, cv=3, scoring='accuracy', n_jobs=-1, verbose=100)
    rf_best2.fit(feature_train, target_train)
    
    rf_param_grid3 = {'max_features': np.arange(5, 81, 5)}
    rf_best3 = GridSearchCV(rf_best2.best_estimator_, param_grid=rf_param_grid3, cv=3, scoring='accuracy', n_jobs=-1, verbose=100)
    rf_best3.fit(feature_train, target_train)
    
    rf_param_grid4 = {'max_depth': np.arange(5, 20, 2)}
    rf_best4 = GridSearchCV(rf_best3.best_estimator_, param_grid=rf_param_grid4, cv=3, scoring='accuracy', n_jobs=-1, verbose=100)
    rf_best4.fit(feature_train, target_train)

    # 输出最佳模型参数
    print(rf_best4.best_estimator_)
    print(rf_best4.best_params_, "  ", "score: ", rf_best4.best_score_)

    return rf_best4.best_params_, rf_best4

# 模型评估
def evaluation(model, feature_train, target_train, target_true, target_pred):
    cross = cross_val_score(model, feature_train, target_train, scoring='accuracy', n_jobs=-1, cv=3)
    acc = accuracy_score(target_true, target_pred)
    confusion_m = confusion_matrix(target_true, target_pred)
    classes = list(set(target_true))
    classes.sort()
    indices = range(len(confusion_m))

    print('cross_val_score: {}'.format(cross))
    print('accuracy_socre: %.3f' % acc)
    plt.imshow(confusion_m, cmap=plt.cm.Blues)
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('true')
    plt.ylabel('pred')
    for first_index in range(len(confusion_m)):
        for second_index in range(len(confusion_m[first_index])):
            plt.text(first_index, second_index, confusion_m[first_index][second_index], fontsize=12)
    plt.show()

# 模型选择
gbdt_param, gbdt = gbdt_optim(x1_train, y1_train[:, 3])
rf_param, rf = rf_optim(x1_train, y1_train[:, 3])

# 语音通话整体满意度(gbdt score: 0.5800779832810473)
rf11 = RandomForestClassifier(n_estimators=1200,
                              max_depth=13,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=50,
                              n_jobs=-1,
                              random_state=42)# score: 0.5827367210616371
rf11.fit(x1_train, y1_train[:, 0])# 训练模型
y11_train_pred = rf11.predict(x1_train)
y11_val_pred = rf11.predict(x1_val)
print('train score: {}, val score: {}'.format(rf11.score(x1_train, y1_train[:, 0]), rf11.score(x1_val, y1_val[:, 0])))

# 网络覆盖与信号强度(gbdt score: 0.5062399178489068)
rf12 = RandomForestClassifier(n_estimators=800,
                              max_depth=7,
                              criterion='gini',
                              max_features=10,
                              min_samples_split=40,
                              n_jobs=-1,
                              random_state=42)# score: 0.5078755330594126
rf12.fit(x1_train, y1_train[:, 1])
y12_train_pred = rf12.predict(x1_train)
y12_val_pred = rf12.predict(x1_val)
print('train score: {}, val score: {}'.format(rf12.score(x1_train, y1_train[:, 1]), rf12.score(x1_val, y1_val[:, 1])))

# 语音通话清晰度(gbdt score: 0.5491922353156804)
rf13 = RandomForestClassifier(n_estimators=1000,
                              max_depth=13,
                              criterion='gini',
                              max_features=10,
                              min_samples_split=70,
                              n_jobs=-1,
                              random_state=42)# score: 0.5518521029248752
rf13.fit(x1_train, y1_train[:, 2])
y13_train_pred = rf13.predict(x1_train)
y13_val_pred = rf13.predict(x1_val)
print('train score: {}, val score: {}'.format(rf13.score(x1_train, y1_train[:, 2]), rf13.score(x1_val, y1_val[:, 2])))

# 语音通话稳定性(gbdt score: 0.5228064691475192)
rf14 = RandomForestClassifier(n_estimators=1500,
                              max_depth=7,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=35,
                              n_jobs=-1,
                              random_state=42)# score: 0.5236252182766097
rf14.fit(x1_train, y1_train[:, 3])
y14_train_pred = rf14.predict(x1_train)
y14_val_pred = rf14.predict(x1_val)
print('train score: {}, val score: {}'.format(rf14.score(x1_train, y1_train[:, 3]), rf14.score(x1_val, y1_val[:, 3])))


# 手机上网整体满意度(gbdt score: 0.4384298828743273)
rf21 = RandomForestClassifier(n_estimators=1500,
                              max_depth=9,
                              criterion='gini',
                              max_features=10,
                              min_samples_split=75,
                              n_jobs=-1,
                              random_state=42)# score: 0.438588160810383
rf21.fit(x2_train, y2_train[:, 0])
y21_train_pred = rf21.predict(x2_train)
y21_val_pred = rf21.predict(x2_val)
print('train score: {}, val score: {}'.format(rf21.score(x2_train, y2_train[:, 0]), rf21.score(x2_val, y2_val[:, 0])))

# 网络覆盖与信号强度(gbdt score: 0.40440012662234887)
rf22 = RandomForestClassifier(n_estimators=1500,
                              max_depth=15,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=100,
                              n_jobs=-1,
                              random_state=42) # score: 0.4056663501107945
rf22.fit(x2_train, y2_train[:, 1])
y22_train_pred = rf22.predict(x2_train)
y22_val_pred = rf22.predict(x2_val)
print('train score: {}, val score: {}'.format(rf22.score(x2_train, y2_train[:, 1]), rf22.score(x2_val, y2_val[:, 1])))

# 手机上网速度(rf score:  0.39870212092434315)
gbdt23 = GradientBoostingClassifier(max_depth=5,
                                    max_features=11,
                                    min_samples_leaf=70,
                                    min_samples_split=1700,
                                    n_estimators=50,
                                    subsample=0.85,
                                    learning_rate=0.1)# score:  0.3996517885406774
gbdt23.fit(x2_train, y2_train[:, 2])
y23_train_pred = gbdt23.predict(x2_train)
y23_val_pred = gbdt23.predict(x2_val)
print('train score: {}, val score: {}'.format(gbdt23.score(x2_train, y2_train[:, 2]), gbdt23.score(x2_val, y2_val[:, 2])))

# 手机上网稳定性(rf score:  0.3980690091801202)
gbdt24 = GradientBoostingClassifier(max_depth=5,
                                    max_features=11,
                                    min_samples_leaf=70,
                                    min_samples_split=1500,
                                    n_estimators=50,
                                    subsample=0.85,
                                    learning_rate=0.1)# score: 0.40028490028490027
gbdt24.fit(x2_train, y2_train[:, 3])
y24_train_pred = gbdt24.predict(x2_train)
y24_val_pred = gbdt24.predict(x2_val)
print('train score: {}, val score: {}'.format(gbdt24.score(x2_train, y2_train[:, 3]), gbdt24.score(x2_val, y2_val[:, 3])))

if __name__ == '__main__':
    # 各项标签统计
    x_data = np.arange(1, 11, 1)
    x_data = [str(data) for data in x_data]
    for i in range(len(x_data)):
        plt.bar(x_data[i], label_summary(y2[:, 3])[i])
    plt.title('手机上网稳定性')
    plt.xlabel('得分')
    plt.show()

    # 各模型评估
    evaluation(gbdt24, x2_train, y2_train[:, 3], y2_val[:, 3], y24_val_pred)

    # 影响因素权重图
    fi = pd.Series(gbdt24.feature_importances_, index = surfing_head)
    fi = fi.sort_values(ascending=True)
    fig = plt.figure(figsize=(12, 6))
    plt.barh(fi.index, fi.values, color='blue')
    plt.xlabel('Importance')
    plt.yticks(range(len(fi)), fi.index)
    for i, j in zip(range(len(fi)), fi.values):
        plt.text(j + 0.001, i, '%s' % round(j, 3), va='center', fontdict=font)
    plt.show()
