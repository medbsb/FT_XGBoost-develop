import os

from tushare import pro

os.environ['OPENBLAS_NUM_THREADS'] = '1'
from datetime import datetime
import circulatingutil as kai
import time
import tushare as ts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from talib import *
from xpinyin import Pinyin
ts.set_token("49cbb8fa012ee0c2295a1b9da1b6c3a9bab7c45941579767d6daaf40")
pro = ts.pro_api()
def VHF(close):

    LCP = MIN(close, timeperiod=28)
    HCP = MAX(close, timeperiod=28)
    NUM = HCP - LCP
    pre = close.copy()
    pre = pre.shift()
    DEN = abs(close-close.shift())
    DEN = MA(DEN, timeperiod=28)*28
    return NUM.div(DEN)
pd.options.mode.chained_assignment = None  # default='warn'
# zhongxiaoban
data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
data = data[data.market.str.contains("中小板")]
a = data.name.str.contains("ST")
a = [not elem for elem in a]
data = data[a]
print(data)
indicators = ['ma 13-5','macd 12-26','macd_signal 12-26','PPO','ROC','TRIX','WILLR','ULTOSC','quadrature','inphase','ADOSC','vol_r','mfi','OBV','MOM',"NATR"]
print("len of indicatiors",len(indicators))
print("stock number:", len(data['ts_code'][::]))
df = pd.DataFrame()
count = 0
# today info
today = datetime.today().strftime('%Y%m%d')
df_trade_date = pro.trade_cal(exchange='', start_date='20220101', end_date=today)
today = df_trade_date['cal_date'][df_trade_date['is_open']==1].values[0]
print("begin to run, today is :", today)
###############
df_ths_index = pro.ths_index()
#############
for i in range(len(data['ts_code'][::])):
# for i in range(100): #for debug
    time.sleep(0.1)
    i = data['ts_code'].index[::][i]
    count+=1
    outputpath = "/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv"
    try :
        df_basic = pd.read_csv(outputpath + "update_basic_" + str(today) +'_'+ data['ts_code'][i].split('.')[0] +".csv")
    except:
        df_basic = pd.DataFrame()
    # df_basic = ts.pro_bar(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718',adj='qfq')
    if(type(df_basic) == type(None)):continue
    try :
        df_money = pd.read_csv(outputpath + "update_moneyflow_" + str(today) +'_'+ data['ts_code'][i].split('.')[0] +".csv")
    except:
        df_money = pd.DataFrame()
    # df_money = pro.moneyflow(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718')
    df_money = df_money.dropna()
    if(len(df_basic) < 100 or len(df_money) < 100):
        print("To small!!!")
        continue
    # slim 
    start_date = 20220120
    end_date = 20231212
    df_basic = df_basic[(df_basic['trade_date'] > start_date) & (df_basic['trade_date'] <= end_date)]
    df_money = df_money[(df_money['trade_date'] > start_date) & (df_money['trade_date'] <= end_date)]
    df_ths_member = pro.ths_member(code=data['ts_code'][i])
    if(len(df_ths_member)==0):
        ths_name = "wu"
    else:
        ths_name = df_ths_index['name'][df_ths_index['ts_code'] == df_ths_member['ts_code'].values[0]].values[0]
    if(len(df_basic) < 100 or len(df_money) < 100):
        print("To small!!!")
        continue
    df_money['all_vol'] = df_money['buy_sm_vol'] + df_money['buy_md_vol'] + df_money['buy_lg_vol'] + df_money['buy_elg_vol'] + df_money['sell_sm_vol'] + df_money['sell_md_vol'] + df_money['sell_lg_vol'] + df_money['sell_elg_vol']
    df_money['lg_elg_buy_ratio'] = (df_money['buy_lg_vol'] + df_money['buy_elg_vol']) / df_money['all_vol']
    df1 = df_basic
    print("the number :{0} stock start".format(count))
    if(len(df1)<100):
        print("too small")
        continue
    macd, macd_signal, macdhist = MACD(df1.close, fastperiod=12, slowperiod=26, signalperiod=9)
    MA5 = EMA(df1.close,timeperiod=5)
    MA13 = EMA(df1.close,timeperiod=30)
    df1['name'] = data['name'][i]
    df1[indicators[0]] = T3(df1.close.values[::-1],timeperiod=5,vfactor=0)[::-1]
    df1['ths_name'] = Pinyin().get_pinyin(ths_name, tone_marks='marks')
    df1[indicators[0]] = VHF(df1.close)
    # df1[indicators[0] = (MA13 - MA5) / MA5
    macd12 = EMA(macd,timeperiod=12)
    macd26 = EMA(macd,timeperiod=26)
    macd_signal12 = EMA(macd_signal,timeperiod=12)
    macd_signal26 = EMA(macd_signal,timeperiod=26)
    macdhist12 = EMA(macdhist,timeperiod=12)
    macdhist26 = EMA(macdhist,timeperiod=26)
    # df1[indicators[1] = (macd12-macd26)*100/macd26
    # df1[indicators[2] = (macd_signal12-macd_signal26)*100/macd_signal26
    df1[indicators[1]] = macdhist
    df1[indicators[2]] = macd
    # df1[indicators[1] = (macdhist) - (macdhist).shift(-1)
    df1[indicators[3]] = PPO(df1.close, fastperiod=12, slowperiod=26, matype=0)
    df1[indicators[4]] = ROC(df1.close, timeperiod=10)
    slowk,slowd = STOCH(df1['high'],
                                           df1['low'],
                                           df1['close'],
                                           fastk_period=9,
                                           slowk_period=5,
                                           slowk_matype=1,
                                           slowd_period=5,
                                           slowd_matype=1)#计算kdj的正确配置
    # df1[indicators[5] = slowk
    # df1[indicators[6] = slowk *slowd
    fastk, fastd = STOCHRSI(df1.close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    trix = TRIX(df1.close, timeperiod=30)
    trix12 = MA(trix,timeperiod=12)
    trix26 = MA(trix,timeperiod=26)
    df1[indicators[5]] = (trix12 - trix26)*100 / trix26
    # df1[indicators[5]] = TRIX(df1.close, timeperiod=30)
    # df1[indicators[7] = trix - trix.shift(-1)
    willr = WILLR(df1.high, df1.low, df1.close, timeperiod=14)
    willr12 = MA(willr,timeperiod=12)
    willr26 = MA(willr,timeperiod=26)
    df1[indicators[6]] = (willr12 - willr26)*100 / willr26
    # df1[indicators[8] = WILLR(df1.high, df1.low, df1.close, timeperiod=14)
    ultosc = ULTOSC(df1.high, df1.low, df1.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ultosc12 = MA(ultosc,timeperiod=12)
    ultosc26 = MA(ultosc,timeperiod=26)
    df1[indicators[7]] = ultosc
    # df1[indicators[9] = ULTOSC(df1.high, df1.low, df1.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    inphase, quadrature = HT_PHASOR(df1.close)
    df1[indicators[8]] = quadrature 
    df1[indicators[9]] = inphase
    df1[indicators[10]] = ADOSC(df1.high, df1.low, df1.close, df1.vol, fastperiod=3, slowperiod=10)/1000
    vol12 = MA(df1.vol,timeperiod=12)
    vol26 = MA(df1.vol,timeperiod=26)
    df1[indicators[11]] = (vol12-vol26)*100/vol26
    # df1[indicators[12] = df1['vol'] / df1.shift(-1)['vol'].rolling(10).mean()
    mfi = MFI(df1.high,df1.low,df1.close,df1.vol,timeperiod=14)
    df1[indicators[12]] = mfi
    obv = OBV(df1.close,df1.vol)
    obv12 = MA(obv,timeperiod=12)
    obv26 = MA(obv,timeperiod=26)
    df1[indicators[13]] =  (obv12 - obv26)*100 / obv26
    mom = MOM(df1.close, timeperiod=10)
    df1[indicators[14]] = mom
    df1[indicators[15]] = NATR(df1.high, df1.low, df1.close, timeperiod=14)
    print("end {0}".format(count))
    df1 = df1.dropna()
    df = pd.concat([df,df1],axis=0)
    print("the number :{0} stock end".format(count))
import xgboost as xgb
def claim_model(model_whole_path):
    model = xgb.XGBClassifier()
    model.load_model(model_whole_path)
    return model
# add the proba in the dataframe
def get_proba_value(model, df_predict, indicators):
    value = model.predict_proba(df_predict[indicators])
    return value[:,1]

print("start the proba: \n")
indicators = ['ma 13-5','macd 12-26','macd_signal 12-26','PPO','ROC','TRIX','WILLR','ULTOSC','quadrature','inphase','ADOSC','vol_r','mfi','OBV','MOM',"NATR"]
name_1day_chg10_zhongxiaoban = "Sun_Feb__5_1day_chg10_model__zhongxiaoban_new_factors_WP_0.17495257.json"
name_1day_zhongxiaoban = "Sun_Feb__5_1day_model__zhongxiaoban_new_factors_WP_0.037452295.json"
name_2day_zhongxiaoban = "Sun_Feb__5_2day_model__zhongxiaoban_new_factors_WP_0.7383144.json"
name_3day_zhongxiaoban = "Sun_Feb__5_3day_model__zhongxiaoban_new_factors_WP_0.70850486.json"
name_4day_zhongxiaoban = "Sun_Feb__5_4day_model__zhongxiaoban_new_factors_WP_0.6855117.json"
name_5day_zhongxiaoban = "Sun_Feb__5_5day_model__zhongxiaoban_new_factors_WP_0.6779077.json"
name_6day_zhongxiaoban = "Sun_Feb__5_6day_model__zhongxiaoban_new_factors_WP_0.6740984.json"
name_7day_zhongxiaoban = "Sun_Feb__5_7day_model__zhongxiaoban_new_factors_WP_0.66473144.json"
name_8day_zhongxiaoban = "Sun_Feb__5_8day_model__zhongxiaoban_new_factors_WP_0.6714772.json"
name_9day_zhongxiaoban = "Sun_Feb__5_9day_model__zhongxiaoban_new_factors_WP_0.66726303.json"
name_10day_zhongxiaoban = "Sun_Feb__5_10day_model__zhongxiaoban_new_factors_WP_0.6853707.json"
name_20day_zhongxiaoban = "Sun_Feb__5_20day_model__zhongxiaoban_new_factors_WP_0.6497563.json"
name_30day_zhongxiaoban = "Sun_Feb__5_30day_model__zhongxiaoban_new_factors_WP_0.63655764.json"
name_40day_zhongxiaoban = "Sun_Feb__5_40day_model__zhongxiaoban_new_factors_WP_0.6301059.json"
name_50day_zhongxiaoban = "Sun_Feb__5_50day_model__zhongxiaoban_new_factors_WP_0.6589336.json"
name_60day_zhongxiaoban = "Sun_Feb__5_60day_model_zhongxiaoban_WP_0.5847918.json"


model_target_1day_chg10_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_1day_chg10_zhongxiaoban)
model_target_1day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_1day_zhongxiaoban)
model_target_2day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_2day_zhongxiaoban)
model_target_3day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_3day_zhongxiaoban)
model_target_4day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_4day_zhongxiaoban)
model_target_5day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_5day_zhongxiaoban)
model_target_6day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_6day_zhongxiaoban)
model_target_7day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_7day_zhongxiaoban)
model_target_8day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_8day_zhongxiaoban)
model_target_9day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_9day_zhongxiaoban)
model_target_10day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_10day_zhongxiaoban)
model_target_20day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_20day_zhongxiaoban)
model_target_30day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_30day_zhongxiaoban)
model_target_40day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_40day_zhongxiaoban)
model_target_50day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_50day_zhongxiaoban)
model_target_60day_zhongxiaoban = claim_model("/hpcfs/cms/cmsgpu/zhangzhx/test/test_err/xgboost/zhongxiaoban/"+name_60day_zhongxiaoban)

# 中小板
df['proba_1day_chg10'] = get_proba_value(model_target_1day_chg10_zhongxiaoban, df,indicators=indicators)
df['proba_1day'] = get_proba_value(model_target_1day_zhongxiaoban, df,indicators=indicators)
df['proba_2day'] = get_proba_value(model_target_2day_zhongxiaoban, df,indicators=indicators)
df['proba_3day'] = get_proba_value(model_target_3day_zhongxiaoban, df,indicators=indicators)
df['proba_4day'] = get_proba_value(model_target_4day_zhongxiaoban, df,indicators=indicators)
df['proba_5day'] = get_proba_value(model_target_5day_zhongxiaoban, df,indicators=indicators)
df['proba_6day'] = get_proba_value(model_target_6day_zhongxiaoban, df,indicators=indicators)
df['proba_7day'] = get_proba_value(model_target_7day_zhongxiaoban, df,indicators=indicators)
df['proba_8day'] = get_proba_value(model_target_8day_zhongxiaoban, df,indicators=indicators)
df['proba_9day'] = get_proba_value(model_target_9day_zhongxiaoban, df,indicators=indicators)
df['proba_10day'] = get_proba_value(model_target_10day_zhongxiaoban, df,indicators=indicators)
df['proba_20day'] = get_proba_value(model_target_20day_zhongxiaoban, df,indicators=indicators)
df['proba_30day'] = get_proba_value(model_target_30day_zhongxiaoban, df,indicators=indicators)
df['proba_40day'] = get_proba_value(model_target_40day_zhongxiaoban, df,indicators=indicators)
df['proba_50day'] = get_proba_value(model_target_50day_zhongxiaoban, df,indicators=indicators)
df['proba_60day'] = get_proba_value(model_target_60day_zhongxiaoban, df,indicators=indicators)

df.to_csv("xgboost_application_zhongxiaoban.csv",index=False)
