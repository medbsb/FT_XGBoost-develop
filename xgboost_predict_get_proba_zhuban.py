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
from IPython.display import display
pd.options.mode.chained_assignment = None  # default='warn'

ts.set_token("49cbb8fa012ee0c2295a1b9da1b6c3a9bab7c45941579767d6daaf40")
pro = ts.pro_api()
def get_factors(data,today,indicators,debug):
    count = 0
    df_predict = pd.DataFrame()
    df_ths_index = pro.ths_index()
    for i in range(len(data['ts_code'][::])):
        time.sleep(0.1)
        i = data['ts_code'].index[::][i]
        count += 1
        outputpath = "/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/"
        try:
            df_basic = pd.read_csv(
                outputpath + "update_basic_" + str(today) + '_' + data['ts_code'][i].split('.')[0] + ".csv")
            df_basic1 = pro.daily_basic(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718',
                                        fields='ts_code,trade_date,circ_mv,dv_ratio,pb')
        except:
            df_basic = pd.DataFrame()
        # df_basic = ts.pro_bar(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718',adj='qfq')
        if (type(df_basic) == type(None)): continue
        try:
            df_money = pd.read_csv(
                outputpath + "update_moneyflow_" + str(today) + '_' + data['ts_code'][i].split('.')[0] + ".csv")
        except:
            df_money = pd.DataFrame()
        # df_money = pro.moneyflow(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718')
        df_money = df_money.dropna()
        if (len(df_basic) < 100 or len(df_money) < 100):
            print("To small!!!")
            continue
        # slim
        start_date = 20100101
        end_date = 20231212
        df_basic = df_basic[(df_basic['trade_date'] > start_date) & (df_basic['trade_date'] <= end_date)]
        df_money = df_money[(df_money['trade_date'] > start_date) & (df_money['trade_date'] <= end_date)]

        if (len(df_basic) < 100 or len(df_money) < 100):
            print("To small!!!")
            continue
        df_money['all_vol'] = df_money['buy_sm_vol'] + df_money['buy_md_vol'] + df_money['buy_lg_vol'] + df_money[
            'buy_elg_vol'] + df_money['sell_sm_vol'] + df_money['sell_md_vol'] + df_money['sell_lg_vol'] + df_money[
                                  'sell_elg_vol']
        df_money['lg_elg_buy_ratio'] = (df_money['buy_lg_vol'] + df_money['buy_elg_vol']) / df_money['all_vol']
        # attention: df_basic trade_date order is good with old date to new date
        df1 = df_basic
        #df2 = df_basic1
        print("the number :{0} stock start".format(count))
        df1[indicators[0]] = T3(df1.close.values, timeperiod=5, vfactor=0)
        ema = EMA(df1.close.values, timeperiod=30)
        df1[indicators[1]] = ema
        df1[indicators[2]] = HT_TRENDLINE(df1.close.values)
        df1[indicators[3]] = KAMA(df1.close.values, timeperiod=5)
        df1[indicators[4]] = SAR(df1.high.values, df1.low.values, acceleration=0, maximum=0)
        df1[indicators[5]] = ADX(df1.high.values, df1.low.values, df1.close.values, timeperiod=5)
        df1[indicators[6]] = APO(df1.close.values, fastperiod=12, slowperiod=26, matype=0)
        df1[indicators[7]] = AROONOSC(df1.high.values, df1.low, timeperiod=14)
        cci = CCI(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[8]] = cci - cci.shift(-1).fillna(np.mean(cci))

        df1[indicators[9]] = CMO(df1.close, timeperiod=14)
        df1[indicators[10]] = DX(df1.high, df1.low, df1.close, timeperiod=14)
        macd, macd_signal, macdhist = MACD(df1.close, fastperiod=12, slowperiod=26, signalperiod=9)
        df1[indicators[11]] = macd
        df1[indicators[12]] = macd
        df1[indicators[13]] = macd_signal
        macd, macd_signal, macdhist = MACDFIX(df1.close, signalperiod=9)
        df1[indicators[14]] = macd
        df1[indicators[15]] = macd_signal
        df1[indicators[16]] = macdhist
        df1[indicators[17]] = MFI(df1.high, df1.low, df1.close, df1.vol, timeperiod=14)
        df1[indicators[18]] = MINUS_DI(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[19]] = MINUS_DM(df1.high, df1.low, timeperiod=14)
        df1[indicators[20]] = MOM(df1.close, timeperiod=10)
        df1[indicators[21]] = PLUS_DI(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[22]] = PLUS_DM(df1.high, df1.low, timeperiod=14)
        df1[indicators[23]] = PPO(df1.close, fastperiod=12, slowperiod=26, matype=0)
        df1[indicators[24]] = ROC(df1.close, timeperiod=10)
        df1[indicators[25]] = ROCP(df1.close, timeperiod=10)
        df1[indicators[26]] = ROCR(df1.close, timeperiod=10)
        df1[indicators[27]] = ROCR100(df1.close, timeperiod=10)
        rsi = RSI(df1.close, timeperiod=6)
        df1[indicators[28]] = rsi - rsi.shift(-1).rolling(5).mean().fillna(np.mean(rsi))
        slowk, slowd = STOCH(df1['high'],
                             df1['low'],
                             df1['close'],
                             fastk_period=9,
                             slowk_period=5,
                             slowk_matype=1,
                             slowd_period=5,
                             slowd_matype=1)  # 计算kdj的正确配置
        slowj = 3 * slowk - 2 * slowd
        df1[indicators[29]] = slowk
        df1[indicators[30]] = slowd
        df1[indicators[31]] = slowj
        fastk, fastd = STOCHRSI(df1.close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df1[indicators[32]] = fastk
        df1[indicators[33]] = fastk - fastd
        df1[indicators[34]] = TRIX(df1.close, timeperiod=30)
        df1[indicators[35]] = ULTOSC(df1.high, df1.low, df1.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        df1[indicators[36]] = WILLR(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[37]] = AD(df1.high, df1.low, df1.close, df1.vol)
        df1[indicators[38]] = ADOSC(df1.high, df1.low, df1.close, df1.vol, fastperiod=3, slowperiod=10)
        df1[indicators[39]] = HT_DCPERIOD(df1.close)
        df1[indicators[40]] = HT_DCPHASE(df1.close)
        inphase, quadrature = HT_PHASOR(df1.close)
        df1[indicators[41]] = inphase
        df1[indicators[42]] = quadrature
        sine, leadsine = HT_SINE(df1.close)
        df1[indicators[43]] = sine
        df1[indicators[44]] = leadsine
        df1[indicators[45]] = HT_TRENDMODE(df1.close)
        aroondown, aroonup = AROON(df1.high, df1.low, timeperiod=14)
        df1[indicators[46]] = aroondown
        df1[indicators[47]] = aroonup
        df1[indicators[48]] = aroonup - aroondown
        df1[indicators[49]] = BOP(df1.open, df1.high, df1.low, df1.close)
        df1[indicators[50]] = ATR(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[51]] = NATR(df1.high, df1.low, df1.close, timeperiod=14)
        df1[indicators[52]] = TRANGE(df1.high, df1.low, df1.close)
        vol10 = MA(df1.vol, timeperiod=10)
        df1[indicators[53]] = df1.vol / vol10
        # df1[indicators[12] = df1['vol'] / df1.shift(-1)['vol'].rolling(10).mean()
        mfi = MFI(df1.high, df1.low, df1.close, df1.vol, timeperiod=14)
        df1[indicators[54]] = mfi
        mom = MOM(df1.close, timeperiod=10)
        df1[indicators[55]] = mom
        lg_elg_buy10 = MA(df_money['lg_elg_buy_ratio'], timeperiod=10)
        lg_elg_buy30 = MA(df_money['lg_elg_buy_ratio'], timeperiod=30)
        df1[indicators[56]] = df_money['buy_sm_vol'] / df_money['all_vol']
        df1[indicators[57]] = df_money['buy_md_vol'] / df_money['all_vol']
        df1[indicators[58]] = df_money['buy_lg_vol'] / df_money['all_vol']
        df1[indicators[59]] = df_money['buy_elg_vol'] / df_money['all_vol']
        df1[indicators[60]] = df_money['sell_sm_vol'] / df_money['all_vol']
        df1[indicators[61]] = df_money['sell_md_vol'] / df_money['all_vol']
        df1[indicators[62]] = df_money['sell_lg_vol'] / df_money['all_vol']
        df1[indicators[63]] = df_money['sell_elg_vol'] / df_money['all_vol']
        MTM, MTMMA = kai.MTM(df1.close)
        df1[indicators[64]] = MTM
        df1[indicators[65]] = df1['circ_mv']
        df1[indicators[66]] = df1['dv_ratio']
        df1[indicators[67]] = df1['pb']

        df1 = df1.dropna()
        if(debug):
            print("the end")
            display(df1)
        df_predict = pd.concat([df_predict,df1],axis=0)
        print("the number :{0} stock end".format(count))
    return df_predict
if __name__ == "__main__":
    import time
    pd.options.mode.chained_assignment = None  # default='warn'
    ##today
    # today = datetime.today().strftime('%Y%m%d')
    today = '20230605'
    print("datetime today",today)
    df_trade_date = pro.trade_cal(exchange='', start_date='20220101', end_date=today)
    today = df_trade_date['cal_date'][df_trade_date['is_open']==1].values[0]
    print("begin to run, today is :", today)
    ###
    ### indicators
    indicators = ['T3', 'EMA', 'HT_TRENDLINE', 'KAMA', 'SAR', 'ADX', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'macd',
                  'macd_signal', 'macd_hist', 'fix_macd', 'fix_macd_signal', 'fix_macd_hist', 'MFI', 'MINUS_DI',
                  'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'slowk',
                  'slowd', 'slowj', 'fastk', 'fastd', 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'HT_DCPERIOD',
                  'HT_DCPHASE', 'inphase', 'quadrature', 'sine', 'leadsine', 'HT_TRENDMODE', 'aroondown', 'aroonup',
                  'arroon_dif', 'bop', 'ATR', 'NATR', 'TRANGE', 'vol_r', 'mfi', 'MOM', 'buy_sm_vol', 'buy_md_vol',
                  'buy_lg_vol', 'buy_elg_vol', 'sell_sm_vol', 'sell_md_vol', 'sell_lg_vol', 'sell_elg_vol', 'MTM',
                  'circ_mv', 'dv_ratio', 'pb']
    print("len of indicatiors",len(indicators))
    ####
    ### zhuban
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
    data = data[data.market.str.contains("主板")]
    a = data.name.str.contains("ST")
    a = [not elem for elem in a]
    data = data[a]
    df_ths_index = pro.ths_index()
    print("stock number:", len(data['ts_code'][::]))
    df_zhuban = get_factors(data=data, today=today, indicators=indicators,debug=False)

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

    name_1day_chg10_zhuban = today + '_1day_chg10_model__zhuban_new_indicators_.json'
    name_1day_zhuban = today + '_1day_model__zhuban_new_indicators_.json'
    name_2day_zhuban = today + '_2day_model__zhuban_new_indicators_.json'
    name_3day_zhuban = today + '_3day_model__zhuban_new_indicators_.json'
    name_4day_zhuban = today + '_4day_model__zhuban_new_indicators_.json'
    name_5day_zhuban = today + '_5day_model__zhuban_new_indicators_.json'
    name_6day_zhuban = today + '_6day_model__zhuban_new_indicators_.json'
    name_7day_zhuban = today + '_7day_model__zhuban_new_indicators_.json'
    name_8day_zhuban = today + '_8day_model__zhuban_new_indicators_.json'
    name_9day_zhuban = today + '_9day_model__zhuban_new_indicators_.json'
    name_10day_zhuban = today + '_10day_model__zhuban_new_indicators_.json'
    name_20day_zhuban = today + '_20day_model__zhuban_new_indicators_.json'
    name_30day_zhuban = today + '_30day_model__zhuban_new_indicators_.json'
    name_40day_zhuban = today + '_40day_model__zhuban_new_indicators_.json'
    name_50day_zhuban = today + '_50day_model__zhuban_new_indicators_.json'
    name_60day_zhuban = today + '_60day_model__zhuban_new_indicators_.json'
    # name_1day_chg10_zhuban = 'Tue_Dec__6_1day_chg10_model_zhuban_WP_0.9893033.json'
    # name_1day_zhuban = 'Tue_Dec__6_1day_model_zhuban_WP_0.9671323.json'
    # name_2day_zhuban = 'Tue_Dec__6_2day_model_zhuban_WP_0.98546666.json'
    # name_3day_zhuban = 'Tue_Dec__6_3day_model_zhuban_WP_0.98796684.json'
    # name_4day_zhuban = 'Tue_Dec__6_4day_model_zhuban_WP_0.98737854.json'
    # name_5day_zhuban = 'Tue_Dec__6_5day_model_zhuban_WP_0.9877415.json'
    # name_6day_zhuban = 'Tue_Dec__6_6day_model_zhuban_WP_0.9881165.json'
    # name_7day_zhuban = 'Tue_Dec__6_7day_model_zhuban_WP_0.9885035.json'
    # name_8day_zhuban = 'Tue_Dec__6_8day_model_zhuban_WP_0.9890548.json'
    # name_9day_zhuban = 'Tue_Dec__6_9day_model_zhuban_WP_0.98874205.json'
    # name_10day_zhuban = 'Wed_Dec_21_01:01:50_10day_model_zhuban_WP_0.64531505.json'
    # name_20day_zhuban = 'Tue_Dec__6_20day_model_zhuban_WP_0.98887944.json'
    # name_30day_zhuban = 'Tue_Dec__6_30day_model_zhuban_WP_0.9897596.json'
    # name_40day_zhuban = 'Tue_Dec__6_40day_model_zhuban_WP_0.98981035.json'
    # name_50day_zhuban = 'Tue_Dec__6_50day_model_zhuban_WP_0.9896296.json'
    # name_60day_zhuban = 'Tue_Dec__6_60day_model_zhuban_WP_0.9898075.json'

    model_target_1day_chg10_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_1day_chg10_zhuban)
    model_target_1day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_1day_zhuban)
    model_target_2day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_2day_zhuban)
    model_target_3day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_3day_zhuban)
    model_target_4day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_4day_zhuban)
    model_target_5day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_5day_zhuban)
    model_target_6day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_6day_zhuban)
    model_target_7day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_7day_zhuban)
    model_target_8day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_8day_zhuban)
    model_target_9day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_9day_zhuban)
    model_target_10day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_10day_zhuban)
    model_target_20day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_20day_zhuban)
    model_target_30day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_30day_zhuban)
    model_target_40day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_40day_zhuban)
    model_target_50day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_50day_zhuban)
    model_target_60day_zhuban = claim_model("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/circ_mv_training/"+name_60day_zhuban)

    # 主板
    df_zhuban['proba_1day_chg10'] = get_proba_value(model_target_1day_chg10_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_1day'] = get_proba_value(model_target_1day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_2day'] = get_proba_value(model_target_2day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_3day'] = get_proba_value(model_target_3day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_4day'] = get_proba_value(model_target_4day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_5day'] = get_proba_value(model_target_5day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_6day'] = get_proba_value(model_target_6day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_7day'] = get_proba_value(model_target_7day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_8day'] = get_proba_value(model_target_8day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_9day'] = get_proba_value(model_target_9day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_10day'] = get_proba_value(model_target_10day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_20day'] = get_proba_value(model_target_20day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_30day'] = get_proba_value(model_target_30day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_40day'] = get_proba_value(model_target_40day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_50day'] = get_proba_value(model_target_50day_zhuban, df_zhuban,indicators=indicators)
    df_zhuban['proba_60day'] = get_proba_value(model_target_60day_zhuban, df_zhuban,indicators=indicators)


    # zhuban
    df_zhuban.to_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csv/csvtoday_zhuban_factors_proba.csv",index=False)