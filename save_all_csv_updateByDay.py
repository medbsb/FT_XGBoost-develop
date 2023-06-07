import subprocess
import os 
import csv
import time
from datetime import datetime
import tushare as ts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from IPython.display import display

from talib import *
ts.set_token("49cbb8fa012ee0c2295a1b9da1b6c3a9bab7c45941579767d6daaf40")
pro = ts.pro_api()
pd.options.mode.chained_assignment = None  # default='warn'

def save_all(outputpath):
    '''save all of the stock to csv upto newest date'''
    print("running __save_all__")
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
    print(data)
    a = data.name.str.contains("ST")
    a = [not elem for elem in a]
    data = data[a]
    df = pd.DataFrame()
    count = 0
    for i in range(len(data['ts_code'][::])):
    # for i in range(2):
        i = data['ts_code'].index[::][i]
        print("data ts_code index is: ", i)
        count+=1
        df_basic = ts.pro_bar(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718',adj='qfq')
        df_basic1 = pro.daily_basic(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718',
                                    fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv, circ_mv, free_share, total_share, dv_ttm, dv_ratio, ps_ttm, ps')
        try:
            df_basic = pd.merge(df_basic, df_basic1,on=['ts_code', 'trade_date'])
        except:
            pass

        df_money = pro.moneyflow(ts_code=data['ts_code'][i], start_date='20100101', end_date='20240718')
        if(type(df_basic) == type(None)):continue
        # pass new stock
        if( (len(df_basic) < 1) | (len(df_money)<1) ):
            print("less than 1") 
            continue
        # save with older dates in the first column
        df_basic = df_basic[::-1]
        df_money = df_money[::-1]
        print("saving the file")
        print("latest date is :", df_basic['trade_date'].values[-1])
        print("ts_code is :", data['ts_code'][i].split('.')[0])
        df_basic.to_csv(outputpath + "update_basic_" + str(df_basic['trade_date'].values[-1]) +'_'+ data['ts_code'][i].split('.')[0] +".csv", index=False)
        df_money.to_csv(outputpath + "update_moneyflow_" + str(df_money['trade_date'].values[-1]) +'_'+ data['ts_code'][i].split('.')[0] +".csv", index=False)

def save_one_stock(code_name, outputpath):
    '''save one of the stock to csv upto newest date'''
    print("running __save_one_stock__")
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
    print(data)

    df_basic = ts.pro_bar(ts_code=code_name, start_date='20080101', end_date='20240718',adj='qfq')
    df_money = pro.moneyflow(ts_code=code_name, start_date='20080101', end_date='20240718')
    if(type(df_basic) == type(None)): pass
    # pass new stock
    if( (len(df_basic) < 1) | (len(df_money)<1) ):
        print("less than 1") 
        pass
    # save with older dates in the first column
    df_basic = df_basic[::-1]
    df_money = df_money[::-1]
    print("saving the file")
    print("latest date is :", df_basic['trade_date'].values[-1])
    print("ts_code is :", code_name.split('.')[0])
    df_basic.to_csv(outputpath + "update_basic_" + str(df_basic['trade_date'].values[-1]) +'_'+ code_name.split('.')[0] +".csv", index=False)
    df_money.to_csv(outputpath + "update_moneyflow_" + str(df_money['trade_date'].values[-1]) +'_'+ code_name.split('.')[0] +".csv", index=False)
def add_new_dates(file_date_name,end_date_name):
    # 读取CSV文件
    data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,market,name')
    print(data)
    a = data.name.str.contains("ST")
    a = [not elem for elem in a]
    data = data[a]
    count=0
    for i in range(len(data['ts_code'][::])):
    # for i in range(2):
        count+=1
        i = data['ts_code'].index[::][i]
        #### update basic
        stock_code = data['ts_code'][i]
        code_name = data['ts_code'][i].split('.')[0]
        print("this is the %d one \n " %count)
        try:
            df_basic = pd.read_csv('/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_basic_'+ str(file_date_name)  + '_'+ code_name + '.csv')

            # let the ts_code column to equal the ts_code_x if ts_code_x exists
            df_basic['ts_code'] = df_basic['ts_code'] if 'ts_code_x' in df_basic.columns else df_basic['ts_code']
            #ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_mf_vol,net_mf_amount
            # filter df_basic only save the columns we need : ts_code_x,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv,free_share,total_share,dv_ttm,dv_ratio,ps_ttm,ps
            df_basic = df_basic[['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount','turnover_rate','volume_ratio','pe','pb','total_mv','circ_mv','free_share','total_share','dv_ttm','dv_ratio','ps_ttm','ps']]
            print(df_basic)
            df_moneyflow = pd.read_csv('/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_moneyflow_'+ str(file_date_name)  + '_'+ code_name + '.csv')
        except:
            df_basic = pd.DataFrame()
            df_moneyflow = pd.DataFrame()
            print("file not found %s"%(str(file_date_name) + '_' + str(code_name)))
        if (len(df_basic) ==0 | len(df_moneyflow) ==0 ): continue
        # 获取最新的行情数据
        start_date = str(df_basic['trade_date'].max())  # 取原数据中最大日期作为起始日期
        end_date = end_date_name  # 这里假设您想获取到2022年12月20日的数据
        new_df = ts.pro_bar(ts_code=stock_code, start_date=start_date, end_date=end_date, adj='qfq')
        # add daily basic and merge it with new_df
        new_df_db = pro.daily_basic(ts_code=stock_code, start_date=start_date, end_date=end_date, fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv, circ_mv, free_share, total_share, dv_ttm, dv_ratio, ps_ttm, ps')
        new_df = pd.merge(new_df, new_df_db, on=['trade_date','ts_code'])
        new_df = new_df[::-1]
        new_df = new_df[1::] # 不要已有那个
        # 将新数据合并到原数据中
        df_basic = pd.concat([df_basic, new_df],ignore_index=True)
        df_basic = df_basic[['ts_code','trade_date','open','high','low','close','pre_close','change','pct_chg','vol','amount','turnover_rate','volume_ratio','pe','pb','total_mv','circ_mv','free_share','total_share','dv_ttm','dv_ratio','ps_ttm','ps']]

        # 写入CSV文件
        df_basic.to_csv('/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_basic_'+ str(int(end_date_name))  + '_'+ code_name + '.csv', index=False)
        print("showing the latest df_basic")
        display(df_basic[-3::])
        print("rm the old one with name %s, date is : %s "%(str(code_name),str(file_date_name)))
        print("create the new one with name %s, date is : %s "%(str(code_name),str(end_date_name)))
        # linux command to rm the old one
        # command = 'rm ' + '/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_basic_'+ file_date_name  + '_'+ code_name + '.csv'
        # windows command to rm the old one
        command = 'del ' + 'D:\\pycharm\\finance_spider_data_analysis-main\\FT_XGBoost-develop\\all_csv\\update_basic_'+ file_date_name  + '_'+ code_name + '.csv'
        subprocess.call(command, shell=True)
        #### update moneyflow

        new_df = pro.moneyflow(ts_code=stock_code, start_date=start_date, end_date=end_date)
        new_df = new_df[::-1]
        new_df = new_df[1::] # 不要已有那个
        # 将新数据合并到原数据中
        df_moneyflow = pd.concat([df_moneyflow, new_df],ignore_index=True)
        # 写入CSV文件
        df_moneyflow.to_csv('/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_moneyflow_'+ str(int(end_date_name))  + '_'+ code_name + '.csv', index=False)
        print("showing the latest df_moneyflow")
        display(df_moneyflow[-3::])
        print("rm the old one with name %s, date is : %s "%(str(code_name),str(file_date_name)))
        print("create the new one with name %s, date is : %s "%(str(code_name),str(end_date_name)))
        #Windows
        command = 'del ' + 'D:\\pycharm\\finance_spider_data_analysis-main\\FT_XGBoost-develop\\all_csv\\update_moneyflow_'+ file_date_name  + '_'+ code_name + '.csv'
        #lINUX
        # command = 'rm ' + '/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/update_basic_' + file_date_name + '_' + code_name + '.csv'
        subprocess.call(command, shell=True)
def check_today_info():
    '''check if the dataframe is already or not'''
    new_df = ts.pro_bar(ts_code='000001.SZ', start_date='20221220', end_date='20241202', adj='qfq')
    display(new_df)
    new_df = pro.moneyflow(ts_code='000001.SZ', start_date='20221220', end_date='20241202')
    display(new_df)

if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--function", type=str, required=True, help="Specify which function to run")
    # parser.add_argument("--file_date", type=str, required=False, help="Specify a parameter for the function")
    # parser.add_argument("--real_date", type=str, required=False, help="Specify another parameter for the function")
    # args = parser.parse_args()
    # if args.function == "check_today_info":
    #     check_today_info()
    # elif args.function == "add_new_dates":
    #     add_new_dates(file_date_name = args.file_date, end_date_name = args.real_date)
    # elif args.function == "save_one_stock":
    #     save_one_stock(code_name="000001.SZ", outputpath="/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv")
    # else:
    #     raise ValueError("Invalid function name")

    add_new_dates(file_date_name='20230606', end_date_name='20230607')
    # check_today_info(end_date='20221227')
    # save_all(outputpath="/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/")
    #save_one_stock(outputpath="/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/", code_name = "600663.SH")
    #add_new_day(outputpath="/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/all_csv/",real_today='20230602')
