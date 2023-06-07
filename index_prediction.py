# ---------------------------------- import ---------------------------------- #
import os 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import csv
import time
from datetime import datetime
import tushare as ts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
ts.set_token("49cbb8fa012ee0c2295a1b9da1b6c3a9bab7c45941579767d6daaf40")
pro = ts.pro_api()
from talib import *
from xpinyin import Pinyin
# import circulatingutil as kai
import time
# ------------------------------------ --- ----------------------------------- #
# ------------------------------ load dataframe ------------------------------ #

df_zhuban = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csv/csvtoday_zhuban_factors_proba.csv")
# df_chuangye = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvtoday_chuangye_factors_proba.csv")
# df_zhongxiaoban = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvtoday_zhongxiaoban_factors_proba.csv")
df_zhuban['ts_code'] = df_zhuban['ts_code']
# df_chuangye['ts_code'] = df_chuangye['ts_code_x']
# df_zhongxiaoban['ts_code'] = df_zhongxiaoban['ts_code_x']
# df_all = pd.concat([df_zhuban,df_chuangye,df_zhongxiaoban])
print("finish reading the file")
# ------------------------------------ -- ------------------------------------ #
import datetime
today = datetime.datetime.today()
today_str = today.strftime("%Y%m%d")
# today = int(today_str)
today='20230602'
print("today is" ,today)
df_trade_date = pro.trade_cal(exchange='', start_date='20210101', end_date=today)
today = int(df_trade_date['cal_date'][df_trade_date['is_open']==1].values[0])
yesterday = int(df_trade_date['cal_date'][df_trade_date['is_open']==1].values[1])
the_day_before_yesterday = int(df_trade_date['cal_date'][df_trade_date['is_open']==1].values[2])
# ------------------------------------ -- ------------------------------------ #
# ----------------------------- function to plot ----------------------------- #
def plot_proba(df_predict, today, yesterday, the_day_before_yesterday, sample):
    print("today is %d, yesterday is %d, the day before yesterday is %d"%(today,yesterday,the_day_before_yesterday))

    down_proba_1day_chg10 = 0.05
    down_proba_1day = 0.6
    down_proba_2day = 0.23
    down_proba_3day = 0.2
    down_proba_4day = 0.15
    down_proba_5day = 0.11
    down_proba_6day = 0.1
    down_proba_7day = 0.1
    down_proba_8day = 0.08
    down_proba_9day = 0.07
    down_proba_10day = 0.07
    #取样本下跌最严重的一天
    down_proba_20day = len(df_predict[(df_predict['proba_20day']>0.5) &  (df_predict['trade_date'] == int(20230306))])/len(df_predict[(df_predict['trade_date'] == 20230306)])
    down_proba_30day = len(df_predict[(df_predict['proba_30day']>0.5) &  (df_predict['trade_date'] == int(20230306))])/len(df_predict[(df_predict['trade_date'] == 20230306)])
    down_proba_40day = len(df_predict[(df_predict['proba_40day']>0.5) &  (df_predict['trade_date'] == int(20230306))])/len(df_predict[(df_predict['trade_date'] == 20230306)])
    down_proba_50day = len(df_predict[(df_predict['proba_50day']>0.5) &  (df_predict['trade_date'] == int(20230306))])/len(df_predict[(df_predict['trade_date'] == 20230306)])
    down_proba_60day = len(df_predict[(df_predict['proba_60day']>0.5) &  (df_predict['trade_date'] == int(20230306))])/len(df_predict[(df_predict['trade_date'] == 20230306)])

    up_proba_1day_chg10 = 0.08
    up_proba_1day = 0.7
    up_proba_2day = 0.3
    up_proba_3day = 0.23
    up_proba_4day = 0.2
    up_proba_5day = 0.18
    up_proba_6day = 0.15
    up_proba_7day = 0.15
    up_proba_8day = 0.18
    up_proba_9day = 0.15
    up_proba_10day = 0.15
    #取样本上涨最严重的一天
    up_proba_20day = len(df_predict[(df_predict['proba_20day']>0.5) &  (df_predict['trade_date'] == int(20230112))])/len(df_predict[(df_predict['trade_date'] == 20230112)])
    up_proba_30day = len(df_predict[(df_predict['proba_30day']>0.5) &  (df_predict['trade_date'] == int(20230112))])/len(df_predict[(df_predict['trade_date'] == 20230112)])
    up_proba_40day = len(df_predict[(df_predict['proba_40day']>0.5) &  (df_predict['trade_date'] == int(20230112))])/len(df_predict[(df_predict['trade_date'] == 20230112)])
    up_proba_50day = len(df_predict[(df_predict['proba_50day']>0.5) &  (df_predict['trade_date'] == int(20230112))])/len(df_predict[(df_predict['trade_date'] == 20230112)])
    up_proba_60day = len(df_predict[(df_predict['proba_60day']>0.5) &  (df_predict['trade_date'] == int(20230112))])/len(df_predict[(df_predict['trade_date'] == 20230112)])

    print(len(df_predict[(df_predict['trade_date'] == today)]))
    day3_proba_1day = len(df_predict[(df_predict['proba_1day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_2day = len(df_predict[(df_predict['proba_2day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_3day = len(df_predict[(df_predict['proba_3day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_4day = len(df_predict[(df_predict['proba_4day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_5day = len(df_predict[(df_predict['proba_5day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_6day = len(df_predict[(df_predict['proba_6day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_7day = len(df_predict[(df_predict['proba_7day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_8day = len(df_predict[(df_predict['proba_8day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_9day = len(df_predict[(df_predict['proba_9day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_10day = len(df_predict[(df_predict['proba_10day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_20day = len(df_predict[(df_predict['proba_20day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_30day = len(df_predict[(df_predict['proba_30day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_40day = len(df_predict[(df_predict['proba_40day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_50day = len(df_predict[(df_predict['proba_50day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_60day = len(df_predict[(df_predict['proba_60day']>0.5) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day3_proba_1day_chg10 = len(df_predict[(df_predict['proba_1day_chg10']>0.3) & (df_predict['trade_date'] == the_day_before_yesterday) ])/len(df_predict[(df_predict['trade_date'] == the_day_before_yesterday)])
    day2_proba_1day = len(df_predict[(df_predict['proba_1day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_2day = len(df_predict[(df_predict['proba_2day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_3day = len(df_predict[(df_predict['proba_3day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_4day = len(df_predict[(df_predict['proba_4day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_5day = len(df_predict[(df_predict['proba_5day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_6day = len(df_predict[(df_predict['proba_6day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_7day = len(df_predict[(df_predict['proba_7day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_8day = len(df_predict[(df_predict['proba_8day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_9day = len(df_predict[(df_predict['proba_9day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_10day = len(df_predict[(df_predict['proba_10day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_20day = len(df_predict[(df_predict['proba_20day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_30day = len(df_predict[(df_predict['proba_30day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_40day = len(df_predict[(df_predict['proba_40day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_50day = len(df_predict[(df_predict['proba_50day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_60day = len(df_predict[(df_predict['proba_60day']>0.5) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day2_proba_1day_chg10 = len(df_predict[(df_predict['proba_1day_chg10']>0.3) & (df_predict['trade_date'] == yesterday) ])/len(df_predict[(df_predict['trade_date'] == yesterday)])
    day1_proba_1day = len(df_predict[(df_predict['proba_1day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_2day = len(df_predict[(df_predict['proba_2day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_3day = len(df_predict[(df_predict['proba_3day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_4day = len(df_predict[(df_predict['proba_4day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_5day = len(df_predict[(df_predict['proba_5day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_6day = len(df_predict[(df_predict['proba_6day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_7day = len(df_predict[(df_predict['proba_7day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_8day = len(df_predict[(df_predict['proba_8day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_9day = len(df_predict[(df_predict['proba_9day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_10day = len(df_predict[(df_predict['proba_10day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_20day = len(df_predict[(df_predict['proba_20day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_30day = len(df_predict[(df_predict['proba_30day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_40day = len(df_predict[(df_predict['proba_40day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_50day = len(df_predict[(df_predict['proba_50day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_60day = len(df_predict[(df_predict['proba_60day']>0.5) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    day1_proba_1day_chg10 = len(df_predict[(df_predict['proba_1day_chg10']>0.3) & (df_predict['trade_date'] == today) ])/len(df_predict[(df_predict['trade_date'] == today)])
    plt.figure(figsize=(14,10))
    for i in ['1day_chg10','1day','2day','3day','4day','5day','6day','7day','8day','9day','10day','20day','30day','40day','50day','60day']:
    # for i in ['1day_chg10', '1day', '2day', '3day', '4day', '5day', '6day', '7day', '8day', '9day', '10day']:
        plt.plot(i, eval('up' + '_proba_' + i), color='black', label='up benmark', marker = '^')
        plt.plot(i, eval('down' + '_proba_' + i), color='black', label='down benmark', marker = 'v')
        plt.plot(i, eval('day3' + '_proba_' + i), color='blue', label='-3day', marker = '*')
        plt.plot(i, eval('day2' + '_proba_' + i), color='green', label='-2day', marker = '*')
        plt.plot(i, eval('day1' + '_proba_' + i), color='red', label='-1day', marker = '*')
    # 添加图例
    plt.legend(labels = ['up benmark', 'down benmark', '-3day', '-2day', '-1day'])
    plt.title(sample+"_"+"today is %d, yesterday is %d, the day before yesterday is %d"%(today,yesterday,the_day_before_yesterday))
    # 显示图表
    plt.savefig(str(today)+"_"+str(sample)+".png")
# ------------------------------------ -- ------------------------------------ #
print("start to plot")
plot_proba(df_zhuban,today = int(today), yesterday=yesterday, the_day_before_yesterday=the_day_before_yesterday, sample="zhuban")
# plot_proba(df_chuangye,today = int(today), yesterday=yesterday, the_day_before_yesterday=the_day_before_yesterday,sample="chuangye")
# plot_proba(df_zhongxiaoban,today = int(today), yesterday=yesterday, the_day_before_yesterday=the_day_before_yesterday,sample="zhongxiaoban")
# plot_proba(df_all,today = int(today), yesterday=yesterday, the_day_before_yesterday=the_day_before_yesterday,sample="all")
