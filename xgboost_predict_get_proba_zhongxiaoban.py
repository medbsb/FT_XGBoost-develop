import os

from tushare import pro

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
from talib import *
from xpinyin import Pinyin
import circulatingutil as kai
import time
pd.options.mode.chained_assignment = None  # default='warn'
ts.set_token("49cbb8fa012ee0c2295a1b9da1b6c3a9bab7c45941579767d6daaf40")
pro = ts.pro_api()
def get_factors(data,today,indicators):
    count = 0
    df_predict = pd.DataFrame()
    df_ths_index = pro.ths_index()
    for i in range(len(data['ts_code'][::])):
        i = data['ts_code'].index[::][i]
        count+=1
        try:      
            df1 = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv" + "update_basic_" + str(today) +'_'+ data['ts_code'][i].split('.')[0] +".csv")
            df_money = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv" + "update_moneyflow_" + str(today) +'_'+ data['ts_code'][i].split('.')[0] +".csv")
            df1 = df1[::-1] # turn to tushare timeline
            df_money = df_money[::-1] # turn to tushare timeline
        except:
            df1 = pd.DataFrame()
            df_money = pd.DataFrame()
            print('File is empty')
        if(type(df1) == type(None)):continue
        time.sleep(0.2)
        df_ths_member = pro.ths_member(code=data['ts_code'][i])
        if(len(df_ths_member)==0):
            ths_name = "wu"
        else:
            ths_name = df_ths_index['name'][df_ths_index['ts_code'] == df_ths_member['ts_code'].values[0]].values[0]
        if(len(df1) < 100 or len(df_money) < 100):
            print("To small!!!")
            continue
        df_money['trade_date'] = df_money['trade_date'].astype(np.int64)
        df_money = df_money.dropna()
        # slim the dataframe
        df1 = df1.head(200)
        df_money = df_money.head(200)
        df1 = pd.merge(df1,df_money,how="right",on='trade_date')
        df1['all_vol'] = df1['buy_sm_vol'] + df1['buy_md_vol'] + df1['buy_lg_vol'] + df1['buy_elg_vol'] + df1['sell_sm_vol'] + df1['sell_md_vol'] + df1['sell_lg_vol'] + df1['sell_elg_vol']
        df1['lg_elg_buy_ratio'] = (df1['buy_lg_vol'] + df1['buy_elg_vol']) / df1['all_vol']
        print("the number :{0} stock start".format(count))
        df1['name'] = data['name'][i]
        df1[indicators[0]] = T3(df1.close.values[::-1],timeperiod=5,vfactor=0)[::-1]
        df1['ths_name'] = Pinyin().get_pinyin(ths_name, tone_marks='marks')
        ema = EMA(df1.close.values[::-1],timeperiod=30)[::-1]
        df1[indicators[1]] = ema
        df1[indicators[2]] = HT_TRENDLINE(df1.close.values[::-1])[::-1]
        df1[indicators[3]] = KAMA(df1.close.values[::-1],timeperiod=5)[::-1]
        df1[indicators[4]] = SAR(df1.high.values[::-1],df1.low.values[::-1],acceleration=0,maximum=0)[::-1]
        df1[indicators[5]] = ADX(df1.high.values[::-1],df1.low.values[::-1],df1.close.values[::-1],timeperiod=5)[::-1]
        df1[indicators[6]] = APO(df1.close.values[::-1],fastperiod=12,slowperiod=26,matype=0)[::-1]
        df1[indicators[7]] = AROONOSC(df1.high.values[::-1],df1.low[::-1],timeperiod=14)[::-1]
        cci = CCI(df1.high[::-1],df1.low[::-1],df1.close[::-1],timeperiod=14)[::-1]
        df1[indicators[8]] = cci - cci.shift(-1).fillna(np.mean(cci))

        df1[indicators[9]] = CMO(df1.close[::-1],timeperiod=14)[::-1]
        df1[indicators[10]] = DX(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        macd, macd_signal, macdhist = MACD(df1.close[::-1], fastperiod=12, slowperiod=26, signalperiod=9)[::-1]
        df1[indicators[11]] = macd
        df1[indicators[12]] = macd
        df1[indicators[13]] = macd_signal
        macd, macd_signal, macdhist = MACDFIX(df1.close[::-1], signalperiod=9)[::-1]
        df1[indicators[14]] = macd
        df1[indicators[15]] = macd_signal
        df1[indicators[16]] = macdhist
        df1[indicators[17]] = MFI(df1.high[::-1], df1.low[::-1], df1.close[::-1], df1.vol[::-1], timeperiod=14)[::-1]
        df1[indicators[18]] = MINUS_DI(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        df1[indicators[19]] =  MINUS_DM(df1.high[::-1], df1.low[::-1], timeperiod=14)[::-1]
        df1[indicators[20]] = MOM(df1.close[::-1], timeperiod=10)[::-1]
        df1[indicators[21]] = PLUS_DI(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        df1[indicators[22]] = PLUS_DM(df1.high[::-1], df1.low[::-1], timeperiod=14)[::-1]
        df1[indicators[23]] = PPO(df1.close[::-1], fastperiod=12, slowperiod=26, matype=0)[::-1]
        df1[indicators[24]] = ROC(df1.close[::-1], timeperiod=10)[::-1]
        df1[indicators[25]] = ROCP(df1.close[::-1], timeperiod=10)[::-1]
        df1[indicators[26]] = ROCR(df1.close[::-1], timeperiod=10)[::-1]
        df1[indicators[27]] = ROCR100(df1.close[::-1], timeperiod=10)[::-1]
        rsi = RSI(df1.close[::-1], timeperiod=6)[::-1]
        df1[indicators[28]] = rsi - rsi.shift(-1)[::-1].rolling(5).mean()[::-1].fillna(np.mean(rsi))
        slowk,slowd = STOCH(df1['high'][::-1],
                                               df1['low'][::-1],
                                               df1['close'][::-1],
                                               fastk_period=9,
                                               slowk_period=5,
                                               slowk_matype=1,
                                               slowd_period=5,
                                               slowd_matype=1)#计算kdj的正确配置
        slowj = 3 * slowk - 2 * slowd
        df1[indicators[29]] = slowk
        df1[indicators[30]] = slowd
        df1[indicators[31]] = slowj
        fastk, fastd = STOCHRSI(df1.close[::-1], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[::-1]
        df1[indicators[32]] = fastk
        df1[indicators[33]] = fastk - fastd
        df1[indicators[34]] = TRIX(df1.close[::-1], timeperiod=30)[::-1]
        df1[indicators[35]] = ULTOSC(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod1=7, timeperiod2=14, timeperiod3=28)[::-1]
        df1[indicators[36]] = WILLR(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        df1[indicators[37]] = AD(df1.high[::-1], df1.low[::-1], df1.close[::-1], df1.vol[::-1])[::-1]
        df1[indicators[38]] = ADOSC(df1.high[::-1], df1.low[::-1], df1.close[::-1], df1.vol[::-1], fastperiod=3, slowperiod=10)[::-1]
        df1[indicators[39]] = HT_DCPERIOD(df1.close[::-1])[::-1]
        df1[indicators[40]] =  HT_DCPHASE(df1.close[::-1])[::-1]
        inphase, quadrature = HT_PHASOR(df1.close[::-1])[::-1]
        df1[indicators[41]] = inphase
        df1[indicators[42]] = quadrature
        sine, leadsine = HT_SINE(df1.close[::-1])[::-1]
        df1[indicators[43]] = sine
        df1[indicators[44]] = leadsine
        df1[indicators[45]] = HT_TRENDMODE(df1.close[::-1])[::-1]
        aroondown, aroonup = AROON(df1.high[::-1],df1.low[::-1],timeperiod=14)[::-1]
        df1[indicators[46]] = aroondown
        df1[indicators[47]] = aroonup
        df1[indicators[48]] = aroonup - aroondown
        df1[indicators[49]] = BOP(df1.open[::-1], df1.high[::-1], df1.low[::-1], df1.close[::-1])[::-1]
        df1[indicators[50]] = ATR(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        df1[indicators[51]] = NATR(df1.high[::-1], df1.low[::-1], df1.close[::-1], timeperiod=14)[::-1]
        df1[indicators[52]] = TRANGE(df1.high[::-1], df1.low[::-1], df1.close[::-1])[::-1]
        vol10 = MA(df1.vol[::-1],timeperiod=10)
        df1[indicators[53]] = df1.vol[::-1] / vol10
        # df1[indicators[12] = df1[::-1]['vol'] / df1.shift(-1)[::-1]['vol'].rolling(10).mean()
        mfi = MFI(df1.high[::-1],df1.low[::-1],df1.close[::-1],df1.vol[::-1],timeperiod=14)[::-1]
        df1[indicators[54]] = mfi
        mom = MOM(df1.close[::-1], timeperiod=10)[::-1]
        df1[indicators[55]] = mom
        lg_elg_buy10 = MA(df1['lg_elg_buy_ratio'][::-1],timeperiod=10)
        lg_elg_buy30 = MA(df1['lg_elg_buy_ratio'][::-1],timeperiod=30)
        df1[indicators[56]] =  df1['buy_sm_vol'] / df1['all_vol']
        df1[indicators[57]] =  df1['buy_md_vol'] / df1['all_vol']
        df1[indicators[58]] = df1['buy_lg_vol'] / df1['all_vol']
        df1[indicators[59]] =  df1['buy_elg_vol'] / df1['all_vol']
        df1[indicators[60]] =  df1['sell_sm_vol'] / df1['all_vol']
        df1[indicators[61]] =  df1['sell_md_vol'] / df1['all_vol']
        df1[indicators[62]] =  df1['sell_lg_vol'] / df1['all_vol']
        df1[indicators[63]] =  df1['sell_elg_vol'] / df1['all_vol']
        MTM, MTMMA = kai.MTM(df1.close[::-1])
        df1[indicators[64]] = MTM


        df1 = df1.dropna()
        df_predict = pd.concat([df_predict,df1],axis=0)
        print("the number :{0} stock end".format(count))
    return df_predict
if __name__ == "__main__":
    import time
    pd.options.mode.chained_assignment = None  # default='warn'
    ##today
    # today = datetime.today().strftime('%Y%m%d')
    today = '20230523'
    df_trade_date = pro.trade_cal(exchange='', start_date='20220101', end_date=today)
    today = df_trade_date['cal_date'][df_trade_date['is_open']==1].values[0]
    print("begin to run, today is :", today)
    ###
    ### indicators
    indicators = ['T3','EMA','HT_TRENDLINE','KAMA','SAR','ADX','APO','AROONOSC','CCI','CMO','DX','macd','macd_signal','macd_hist','fix_macd','fix_macd_signal','fix_macd_hist','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DI','PLUS_DM','PPO','ROC','ROCP','ROCR','ROCR100','RSI','slowk','slowd','slowj','fastk','fastd','TRIX','ULTOSC','WILLR','AD','ADOSC','HT_DCPERIOD','HT_DCPHASE','inphase','quadrature','sine','leadsine','HT_TRENDMODE','aroondown','aroonup','arroon_dif','bop','ATR','NATR','TRANGE','vol_r','mfi','MOM','buy_sm_vol','buy_md_vol','buy_lg_vol','buy_elg_vol','sell_sm_vol','sell_md_vol','sell_lg_vol','sell_elg_vol','MTM']
    print("len of indicatiors",len(indicators))
    ####
    # zhongxiaoban
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
    data = data[data.market.str.contains("中小板")]
    a = data.name.str.contains("ST")
    a = [not elem for elem in a]
    data = data[a]
    df_ths_index = pro.ths_index()
    print("stock number:", len(data['ts_code'][::]))
    df_zhongxiaoban = get_factors(data=data, today=today, indicators=indicators)

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

    name_1day_chg10_zhongxiaoban  = "Tue_Dec__6_1day_chg10_model_zhongxiaoban_WP_0.98954344.json"
    name_1day_zhongxiaoban = "Sun_Dec__4_1day_model_zhongxiaoban_WP_0.8557434.json"
    name_2day_zhongxiaoban = "Tue_Dec__6_2day_model_zhongxiaoban_WP_0.98328257.json"
    name_3day_zhongxiaoban = "Tue_Dec__6_3day_model_zhongxiaoban_WP_0.98658556.json"
    name_4day_zhongxiaoban = "Tue_Dec__6_4day_model_zhongxiaoban_WP_0.98808503.json"
    name_5day_zhongxiaoban = "Tue_Dec__6_5day_model_zhongxiaoban_WP_0.9883358.json"
    name_6day_zhongxiaoban = "Tue_Dec__6_6day_model_zhongxiaoban_WP_0.9882009.json"
    name_7day_zhongxiaoban = "Tue_Dec__6_7day_model_zhongxiaoban_WP_0.9883163.json"
    name_8day_zhongxiaoban = "Tue_Dec__6_8day_model_zhongxiaoban_WP_0.98886746.json"
    name_9day_zhongxiaoban = "Tue_Dec__6_9day_model_zhongxiaoban_WP_0.9889826.json"
    name_10day_zhongxiaoban = "Tue_Dec__6_10day_model_zhongxiaoban_WP_0.98867947.json"
    name_20day_zhongxiaoban = "Tue_Dec__6_20day_model_zhongxiaoban_WP_0.98865974.json"
    name_30day_zhongxiaoban = "Tue_Dec__6_30day_model_zhongxiaoban_WP_0.9895256.json"
    name_40day_zhongxiaoban = "Tue_Dec__6_40day_model_zhongxiaoban_WP_0.98920137.json"
    name_50day_zhongxiaoban = "Tue_Dec__6_50day_model_zhongxiaoban_WP_0.98969007.json"
    name_60day_zhongxiaoban = "Tue_Dec__6_60day_model_zhongxiaoban_WP_0.98984927.json"

    model_target_1day_chg10_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_1day_chg10_zhongxiaoban)
    model_target_1day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_1day_zhongxiaoban)
    model_target_2day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_2day_zhongxiaoban)
    model_target_3day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_3day_zhongxiaoban)
    model_target_4day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_4day_zhongxiaoban)
    model_target_5day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_5day_zhongxiaoban)
    model_target_6day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_6day_zhongxiaoban)
    model_target_7day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_7day_zhongxiaoban)
    model_target_8day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_8day_zhongxiaoban)
    model_target_9day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_9day_zhongxiaoban)
    model_target_10day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_10day_zhongxiaoban)
    model_target_20day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_20day_zhongxiaoban)
    model_target_30day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_30day_zhongxiaoban)
    model_target_40day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_40day_zhongxiaoban)
    model_target_50day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_50day_zhongxiaoban)
    model_target_60day_zhongxiaoban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvzhongxiaoban/"+ name_60day_zhongxiaoban)

   
    # 中小板
    df_zhongxiaoban['proba_1day_chg10'] = get_proba_value(model_target_1day_chg10_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_1day'] = get_proba_value(model_target_1day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_2day'] = get_proba_value(model_target_2day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_3day'] = get_proba_value(model_target_3day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_4day'] = get_proba_value(model_target_4day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_5day'] = get_proba_value(model_target_5day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_6day'] = get_proba_value(model_target_6day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_7day'] = get_proba_value(model_target_7day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_8day'] = get_proba_value(model_target_8day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_9day'] = get_proba_value(model_target_9day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_10day'] = get_proba_value(model_target_10day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_20day'] = get_proba_value(model_target_20day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_30day'] = get_proba_value(model_target_30day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_40day'] = get_proba_value(model_target_40day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_50day'] = get_proba_value(model_target_50day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)
    df_zhongxiaoban['proba_60day'] = get_proba_value(model_target_60day_zhongxiaoban, df_zhongxiaoban,indicators=indicators)



    # zhongxiaoban
    df_zhongxiaoban.to_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csvtoday_zhongxiaoban_factors_proba.csv",index=False)
