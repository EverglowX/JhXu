import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 6
pd.set_option('display.max_rows', None, 'display.max_columns', None)
path = 'D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/'
data1 = pd.read_csv(path + '附件1语音业务用户满意度数据.csv', encoding='utf-8')
data2 = pd.read_csv(path + '附件2上网业务用户满意度数据.csv', encoding='utf-8')
data3 = pd.read_csv(path + '附件3语音业务用户满意度预测数据.csv', encoding='utf-8')
data4 = pd.read_csv(path + '附件4上网业务用户满意度预测数据.csv', encoding='utf-8')

# 分别筛选每个附件中需要保留的特征
features1 = ['语音通话整体满意度', '网络覆盖与信号强度', '语音通话清晰度', '语音通话稳定性', '是否遇到过网络问题', '居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明',
            '手机没有信号', '有信号无法拨通', '通话过程中突然中断', '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见','其他，请注明.1', '脱网次数', 'mos质差次数', '未接通掉话次数',
            '是否关怀用户', '套外流量（MB）', '是否4G网络客户（本地剔除物联网）', '套外流量费（元）', '外省语音占比', '语音通话-时长（分钟）', '省际漫游-时长（分钟）', '当月ARPU', '当月MOU',
            '前3月ARPU', '前3月MOU', '外省流量占比', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '是否5G网络客户']
features2 = ['手机上网整体满意度', '网络覆盖与信号强度', '手机上网速度', '手机上网稳定性', '居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号',
            '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢','下载速度慢', '手机支付较慢',
            '其他，请注明.2','爱奇艺', '优酷', '腾讯视频', '芒果TV', '搜狐视频', '抖音', '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿','和平精英', '王者荣耀', '穿越火线',
            '梦幻西游', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ','淘宝', '京东', '百度', '今日头条',
            '新浪微博','拼多多', '其他，请注明.5', '全部网页或APP都慢', '脱网次数', '上网质差次数', '微信质差次数', '套外流量（MB）', '套外流量费（元）','当月MOU', '是否5G网络客户',
            '是否不限量套餐到达用户']
features3 = ['是否遇到过网络问题', '居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁','其他，请注明', '手机没有信号', '有信号无法拨通', '通话过程中突然中断',
            '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见','其他，请注明.1','脱网次数', 'mos质差次数', '未接通掉话次数', '是否关怀用户', '套外流量（MB）',
            '是否4G网络客户（本地剔除物联网）', '套外流量费（元）', '外省语音占比', '语音通话-时长（分钟）','省际漫游-时长（分钟）', '当月ARPU', '当月MOU', '前3月ARPU','前3月MOU',
            '外省流量占比', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '是否5G网络客户']
features4 = ['居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号', '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢',
            '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2','爱奇艺', '优酷', '腾讯视频','芒果TV', '搜狐视频', '抖音',
            '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', '穿越火线', '梦幻西游', '龙之谷', '梦幻诛仙','欢乐斗地主', '部落冲突', '炉石传说', '阴阳师',
            '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博', '拼多多','其他，请注明.5', '全部网页或APP都慢', '脱网次数', '上网质差次数',
            '微信质差次数', '套外流量（MB）', '套外流量费（元）', '当月MOU', '是否5G网络客户', '是否不限量套餐到达用户']

voice_data1 = data1[features1]
surfing_data1 = data2[features2]
voice_data2 = data3[features3]
surfing_data2 = data4[features4]

# 将'是否'数据替换为二进制数据
y_n1 = ['是否关怀用户', '是否4G网络客户（本地剔除物联网）', '是否5G网络客户']
y_n2 = ['是否不限量套餐到达用户', '是否5G网络客户']
for i, name in enumerate(y_n1):
    voice_data1[name].fillna('否', inplace=True)# 将空白处先替换为'否'
    voice_data2[name].fillna('否', inplace=True)
    voice_data1[name] = voice_data1[name].apply(lambda x: x.replace('是', '1').replace('否', '0'))
    voice_data2[name] = voice_data2[name].apply(lambda x: x.replace('是', '1').replace('否', '0'))
    # 将字符型数据转为整型
    voice_data1[name].astype(int)
    voice_data2[name].astype(int)

for i, name in enumerate(y_n2):
    surfing_data1[name].fillna('否', inplace=True)
    surfing_data2[name].fillna('否', inplace=True)
    surfing_data1[name] = surfing_data1[name].apply(lambda x: x.replace('是', '1').replace('否', '0'))
    surfing_data2[name] = surfing_data2[name].apply(lambda x: x.replace('是', '1').replace('否', '0'))
    surfing_data1[name].astype(int)
    surfing_data2[name].astype(int)

# 将百分比数据替换为浮点数
for i in range(len(voice_data1['外省语音占比'])):
    voice_data1.at[i, '外省语音占比'] = float(voice_data1.at[i, '外省语音占比'].strip('%')) / 100.0
for i in range(len(voice_data1['外省流量占比'])):
    # 处理数据中非百分比的特殊值
    if voice_data1.at[i, '外省流量占比'] != '#DIV/0!':
        voice_data1.at[i, '外省流量占比'] = float(voice_data1.at[i, '外省流量占比'].strip('%')) / 100.0
    else:
        voice_data1.at[i, '外省流量占比'] = 0.0

for i in range(len(voice_data2['外省语音占比'])):
    voice_data2.at[i, '外省语音占比'] = float(voice_data2.at[i, '外省语音占比'].strip('%')) / 100.0
for i in range(len(voice_data2['外省流量占比'])):
    if voice_data2.at[i, '外省流量占比'] != '#DIV/0!':
        voice_data2.at[i, '外省流量占比'] = float(voice_data2.at[i, '外省流量占比'].strip('%')) / 100.0
    else:
        voice_data2.at[i, '外省流量占比'] = 0.0

# 缺失值可视化
msno.matrix(voice_data1, labels=True, fontsize=10)
msno.matrix(surfing_data1, labels=True, fontsize=10)
msno.matrix(voice_data2, labels=True, fontsize=10)
msno.matrix(surfing_data2, labels=True, fontsize=10)
plt.show()

# 处理附件中的缺失值
for index, col in voice_data1.items():
    voice_data1[index].fillna(0, inplace=True)
for index, col in voice_data2.items():
    voice_data2[index].fillna(0, inplace=True)
for index, col in surfing_data1.items():
    surfing_data1[index].fillna(0, inplace=True)
for index, col in surfing_data2.items():
    surfing_data2[index].fillna(0, inplace=True)

if __name__ == '__main__':
    # 保存文件
    voice_data1.to_csv('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/q1_voice.csv', sep=',', index=False, header=True)
    surfing_data1.to_csv('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/q1_surfing.csv', sep=',', index=False, header=True)
    voice_data2.to_csv('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/q2_voice.csv', sep=',', index=False, header=True)
    surfing_data2.to_csv('D:/Desktop/File/Machine Learning/MathorCup2022/Track B PreRound/q2_surfing.csv', sep=',', index=False, header=True)
