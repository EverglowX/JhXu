import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import openpyxl
import math

# xlsx文件数据写入
def excel_write(row, col, sheet, data):
    wb = openpyxl.load_workbook('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/result.xlsx')
    ws = wb[sheet]
    for i in range(len(data)):
        ws.cell(row + i, col, data[i])
    wb.save('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/result.xlsx')
    print('File Saved')

np.set_printoptions(threshold=math.inf)
path = 'D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/'
voice_data1 = pd.read_csv(path + 'q1_voice.csv', encoding='utf-8', error_bad_lines=False)
surfing_data1 = pd.read_csv(path + 'q1_surfing.csv', encoding='utf-8', error_bad_lines=False)
voice_data2 = pd.read_csv(path + 'q2_voice.csv', encoding='utf-8', error_bad_lines=False)
surfing_data2 = pd.read_csv(path + 'q2_surfing.csv', encoding='utf-8', error_bad_lines=False)
for index, col in voice_data1.items():
    voice_data1[index].fillna(0, inplace=True)
for index, col in voice_data2.items():
    voice_data2[index].fillna(0, inplace=True)
for index, col in surfing_data1.items():
    surfing_data1[index].fillna(0, inplace=True)
for index, col in surfing_data2.items():
    surfing_data2[index].fillna(0, inplace=True)
voice_data1 = np.array(voice_data1)
voice_data2 = np.array(voice_data2)# x
surfing_data1 = np.array(surfing_data1)
surfing_data2 = np.array(surfing_data2)# x
# 划分特征数据与标签
x1 = voice_data1[:, 4:]
y1 = voice_data1[:, 0: 4].astype(int)
x2 = surfing_data1[:, 4:]
y2 = surfing_data1[:, 0: 4].astype(int)


# 语音通话整体满意度
gbdt11 = GradientBoostingClassifier(max_depth=5,
                                    max_features=9,
                                    min_samples_leaf=60,
                                    min_samples_split=1700,
                                    n_estimators=50,
                                    subsample=0.6,
                                    learning_rate=0.1)
gbdt11.fit(x1, y1[:, 0])
y11_pred = gbdt11.predict(voice_data2)# 获得预测数据

# 网络覆盖与信号强度
gbdt12 = GradientBoostingClassifier(max_depth=10,
                                    max_features=13,
                                    min_samples_leaf=90,
                                    min_samples_split=1700,
                                    n_estimators=50,
                                    subsample=0.6,
                                    learning_rate=0.1)
gbdt12.fit(x1, y1[:, 1])
y12_pred = gbdt12.predict(voice_data2)

# 语音通话清晰度
gbdt13 = GradientBoostingClassifier(max_depth=10,
                                    max_features=9,
                                    min_samples_leaf=100,
                                    min_samples_split=1500,
                                    n_estimators=50,
                                    subsample=0.6,
                                    learning_rate=0.1)
gbdt13.fit(x1, y1[:, 2])
y13_pred = gbdt13.predict(voice_data2)

# 语音通话稳定性
gbdt14 = GradientBoostingClassifier(max_depth=5,
                                    max_features=15,
                                    min_samples_leaf=100,
                                    min_samples_split=1700,
                                    n_estimators=50,
                                    subsample=0.6,
                                    learning_rate=0.1)
gbdt14.fit(x1, y1[:, 3])
y14_pred = gbdt14.predict(voice_data2)


# 手机上网整体满意度
rf21 = RandomForestClassifier(n_estimators=800,
                              max_depth=13,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=60,
                              n_jobs=-1,
                              random_state=42)
rf21.fit(x2, y2[:, 0])
y21_pred = rf21.predict(surfing_data2)

# 网络覆盖与信号强度
rf22 = RandomForestClassifier(n_estimators=200,
                              max_depth=19,
                              criterion='gini',
                              max_features=10,
                              min_samples_split=85,
                              n_jobs=-1,
                              random_state=42)
rf22.fit(x2, y2[:, 1])
y22_pred = rf22.predict(surfing_data2)

# 手机上网速度
rf23 = RandomForestClassifier(n_estimators=1500,
                              max_depth=9,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=95,
                              n_jobs=-1,
                              random_state=42)
rf23.fit(x2, y2[:, 2])
y23_pred = rf23.predict(surfing_data2)

# 手机上网稳定性
rf24 = RandomForestClassifier(n_estimators=400,
                              max_depth=9,
                              criterion='gini',
                              max_features=5,
                              min_samples_split=95,
                              n_jobs=-1,
                              random_state=42)
rf24.fit(x2, y2[:, 3])
y24_pred = rf24.predict(surfing_data2)


if __name__ == '__main__':
    # 将预测结果写入result.xlsx
    excel_write(2, 2, '语音', y11_pred)
    excel_write(2, 3, '语音', y12_pred)
    excel_write(2, 4, '语音', y13_pred)
    excel_write(2, 5, '语音', y14_pred)
    excel_write(2, 2, '上网', y21_pred)
    excel_write(2, 3, '上网', y22_pred)
    excel_write(2, 4, '上网', y23_pred)
    excel_write(2, 5, '上网', y24_pred)
