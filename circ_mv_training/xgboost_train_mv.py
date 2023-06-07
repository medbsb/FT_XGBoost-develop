import pickle
import xgboost as xgb
import numpy as np

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, auc

from sklearn.feature_selection import RFECV, RFE
from functools import partial
import json
import datetime
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import sys
# import ZZTime as zz
#
# zzTime = zz.outputTime()
zzTime = datetime.datetime.now().strftime('%Y%m%d')
print(__name__)

# d0 = pd.read_c0("xgboost_" + day +"s.csv")
def xgb_train(indicators, model_name, colsample_bytree_v, gamma_v, learning_rate_v, max_depth_v, min_child_weight_v,
              estimators_v, reg_alpha_v, reg_lambda_v, subsample_v, Dataframe, target, sample_name, date,
              target_cut_values, is_zhangting):
    print("\n sample is :", sample_name)
    rng = np.random.RandomState(31337)

    print("indicators:", indicators)
    data = Dataframe
    groups = data.groupby('ts_code')
    for name, group in groups:
        # 计算每个分组内开盘价的差异（未来日date -1，date -2）
        diff = (group['open'].shift(-date) - group['open']) / group['open']

        # 将差异结果添加到原始数据框
        data.loc[group.index, 'open_diff'] = diff
    if is_zhangting == True:
        mask = ((data['pct_chg'] + data['pct_chg'].shift(date + 1) + data['pct_chg'].shift(date + 2) + data[
            'pct_chg'].shift(date + 3) + data['pct_chg'].shift(date + 4) + data['pct_chg'].shift(date + 5)) < 0.1)
    else:
        mask = True
    data[target] = np.where((data['open_diff'] > target_cut_values) & (mask), 1, 0)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    traindataset, valdataset = train_test_split(data, test_size=0.3, random_state=rng)
    nS = len(traindataset.iloc[(getattr(traindataset, target).values == 1)])
    nB = len(traindataset.iloc[(getattr(traindataset, target).values == 0)])
    print("nB/nS:", nB / nS)

    model = xgb.XGBClassifier(
        colsample_bytree=colsample_bytree_v, gamma=gamma_v, learning_rate=learning_rate_v, max_depth=max_depth_v,
        min_child_weight=min_child_weight_v, n_estimators=estimators_v, reg_alpha=reg_alpha_v, reg_lambda=reg_lambda_v,
        subsample=subsample_v,
        objective='binary:logistic',
        scale_pos_weight=nB / nS,
        tree_method='gpu_hist'
        )
    print('model \n', model)
    model.fit(
        traindataset[indicators].values,
        getattr(traindataset, target).values,
        eval_set=[(traindataset[indicators].values, getattr(traindataset, target).values),
                  (valdataset[indicators].values, getattr(valdataset, target).values)],
        verbose=True, eval_metric="auc"
    )
    path = "/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/cir_mv_training/"
    model.save_model(path + zzTime + "_" + model_name + "_" + sample_name + ".json")
    # Plot ROC curve
    print("-----plot ROC--------- \n")
    proba = model.predict_proba(traindataset[indicators].values)
    fpr, tpr, thresholds = roc_curve(traindataset[target].values, proba[:, 1])
    train_auc = auc(np.sort(fpr), np.sort(tpr))

    probaT = model.predict_proba(valdataset[indicators].values)
    fprt, tprt, thresholds = roc_curve(valdataset[target].values, probaT[:, 1])
    test_auct = auc(np.sort(fprt), np.sort(tprt))

    fig, ax = plt.subplots(figsize=(6, 6))
    ## ROC curve
    ax.plot(fpr, tpr, lw=1, label='XGB train (area = %0.3f)' % (train_auc))
    ax.plot(fprt, tprt, lw=1, label='XGB test (area = %0.3f)' % (test_auct))
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.grid()
    fig.savefig("roc" + str(zzTime) + "_" + model_name + "_" + sample_name + "_" + ".png")
    plt.cla()
    proba_sig = model.predict_proba(traindataset[indicators][traindataset[target] == 1].values)
    probaT_sig = model.predict_proba(valdataset[indicators][valdataset[target] == 1].values)
    proba_bkg = model.predict_proba(traindataset[indicators][traindataset[target] == 0].values)
    probaT_bkg = model.predict_proba(valdataset[indicators][valdataset[target] == 0].values)
    print("plot binary score plots")
    plt.figure()
    # use for scan threshold
    sig_counts, sig_bins = np.histogram(probaT_sig[:, 1], density=True, bins=200);
    bkg_counts, bkg_bins = np.histogram(probaT_bkg[:, 1], density=True, bins=200);
    # working_point = scan_threshold(sig_counts = sig_counts, sig_bins=sig_bins, bkg_counts= bkg_counts)
    #########
    sig_test_bin_counts, sig_test_bin_edges = np.histogram(probaT_sig[:, 1], density=True, bins=20);
    bkg_test_bin_counts, bkg_test_bin_edges = np.histogram(probaT_bkg[:, 1], density=True, bins=20);
    sig_test_bin_centres = (sig_test_bin_edges[:-1] + sig_test_bin_edges[1:]) / 2
    bkg_test_bin_centres = (bkg_test_bin_edges[:-1] + bkg_test_bin_edges[1:]) / 2
    sig_test_y_error = np.sqrt(sig_test_bin_counts)
    bkg_test_y_error = np.sqrt(bkg_test_bin_counts)
    plt.errorbar(x=sig_test_bin_centres, y=sig_test_bin_counts, yerr=0, fmt='o', capsize=2, label='Sig(test sample)')
    plt.errorbar(x=bkg_test_bin_centres, y=bkg_test_bin_counts, yerr=0, fmt='o', capsize=2, label='Bkgs(test sample)')
    bin_counts, bin_edges, patches = plt.hist(proba_sig[:, 1], density=True, alpha=0.5, label='Sig(training sampel)',
                                              bins=20);
    bin_counts, bin_edges, patches = plt.hist(proba_bkg[:, 1], density=True, alpha=0.5, label='Bkgs(training sample)',
                                              bins=20);
    plt.legend()
    plt.xlabel("BDT score")
    plt.ylabel(r"(1/N) dN/dx")
    plt.savefig("binary_score" + str(zzTime) + "_" + model_name + "_" + sample_name + "_" + ".png")


def scan_threshold(sig_counts, sig_bins, bkg_counts):
    """
    use to scan the better score where sig more than bkg 1/2
    """
    return sig_bins[np.argmin(abs(((sig_counts - bkg_counts) / sig_counts) - 0.5))]


if __name__ == "__main__":
    print("start")
    indicators = ['T3', 'EMA', 'HT_TRENDLINE', 'KAMA', 'SAR', 'ADX', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'macd',
                  'macd_signal', 'macd_hist', 'fix_macd', 'fix_macd_signal', 'fix_macd_hist', 'MFI', 'MINUS_DI',
                  'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'slowk',
                  'slowd', 'slowj', 'fastk', 'fastd', 'TRIX', 'ULTOSC', 'WILLR', 'AD', 'ADOSC', 'HT_DCPERIOD',
                  'HT_DCPHASE', 'inphase', 'quadrature', 'sine', 'leadsine', 'HT_TRENDMODE', 'aroondown', 'aroonup',
                  'arroon_dif', 'bop', 'ATR', 'NATR', 'TRANGE', 'vol_r', 'mfi', 'MOM', 'buy_sm_vol', 'buy_md_vol',
                  'buy_lg_vol', 'buy_elg_vol', 'sell_sm_vol', 'sell_md_vol', 'sell_lg_vol', 'sell_elg_vol', 'MTM',
                  'circ_mv', 'dv_ratio', 'pb']
    # indicators = ['T3','EMA','HT_TRENDLINE','KAMA','SAR','ADX','APO','AROONOSC','CCI','CMO','DX','macd','macd_signal','macd_hist','fix_macd','fix_macd_signal','fix_macd_hist','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DI','PLUS_DM','PPO','ROC','ROCP','ROCR','ROCR100','RSI','slowk','slowd','slowj','fastk','fastd','TRIX','ULTOSC','WILLR','AD','ADOSC','HT_DCPERIOD','HT_DCPHASE','inphase','quadrature','sine','leadsine','HT_TRENDMODE','aroondown','aroonup','arroon_dif','bop','ATR','NATR','TRANGE','vol_r','mfi','MOM','buy_sm_vol','buy_md_vol','buy_lg_vol','buy_elg_vol','sell_sm_vol','sell_md_vol','sell_lg_vol','sell_elg_vol','MTM']
    print(len(indicators))
    # indicators = ['ma 13-5','macd 12-26','macd_signal 12-26','PPO','ROC','TRIX','WILLR','ULTOSC','quadrature','inphase','ADOSC','vol_r','mfi','OBV','MOM',"NATR"]
    df = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csv/xgboost_train_zhuban.csv")

    day = "1day_chg10"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day,
              colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391,
              estimators_v= 4500, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=1, target_cut_values=0.01, is_zhangting=True)
    day = "1day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day,
              colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391,
              estimators_v= 4500, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=1, target_cut_values=0.01, is_zhangting=True)
    day = "2day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day,
              colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391,
              estimators_v= 4500, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=2, target_cut_values=0.02, is_zhangting=True)
    day = "3day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=3, target_cut_values=0.03, is_zhangting=True)
    day = "4day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=4, target_cut_values=0.04, is_zhangting=True)
    day = "5day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=5, target_cut_values=0.05, is_zhangting=True)
    day = "6day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=6, target_cut_values=0.06, is_zhangting=True)
    day = "7day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=7, target_cut_values=0.07, is_zhangting=True)
    day = "8day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=8, target_cut_values=0.08, is_zhangting=True)
    day = "9day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=9, target_cut_values=0.09, is_zhangting=True)
    day = "10day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=10, target_cut_values=0.1, is_zhangting=True)
    day = "20day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=20, target_cut_values=0.2, is_zhangting=True)
    day = "30day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=30, target_cut_values=0.3, is_zhangting=True)
    day = "40day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=40, target_cut_values=0.4, is_zhangting=True)
    day = "50day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=50, target_cut_values=0.5, is_zhangting=True)
    day = "60day"
    xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149,
              gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 4, min_child_weight_v= 0.402676724513391, estimators_v= 4500, reg_alpha_v= 0.587806341330127,
              reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134, date=60, target_cut_values=0.6, is_zhangting=True)

    ### negative target
    # day = "1day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "2day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "3day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "4day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "5day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "6day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "7day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "8day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "9day_neg"
    # xgb_train(indicators=indicators, model_name=  day +"_model",sample_name='_zhuban_new_indicators_', Dataframe=df, target="target_" + day, colsample_bytree_v= 0.8834044009405149, gamma_v= 0.406489868843162, learning_rate_v= 0.080113231069171437, max_depth_v= 3, min_child_weight_v= 0.402676724513391, estimators_v= 10000, reg_alpha_v= 0.587806341330127, reg_lambda_v= 0.366821811291432, subsample_v= 0.879353494846134)
    # day = "2day_0.08"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=0.406489868843162,
    #           learning_rate_v=0.0080113231069171437, max_depth_v=3, min_child_weight_v=0.402676724513391,
    #           estimators_v=10000, reg_alpha_v=0.587806341330127, reg_lambda_v=0.366821811291432,
    #           subsample_v=0.879353494846134, date=2, target_cut_values=0.08, is_zhangting=True)
    #
    # print(day + " done")
    # day = "3day_0.15"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=0.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=3, min_child_weight_v=3.402676724513391,
    #           estimators_v=10000, reg_alpha_v=0.587806341330127, reg_lambda_v=0.366821811291432,
    #           subsample_v=0.879353494846134, date=3, target_cut_values=0.15, is_zhangting=True)
    # print(day + " done")
    #
    # day = "4day_0.3"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=0.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=3, min_child_weight_v=3.402676724513391,
    #           estimators_v=10000, reg_alpha_v=0.587806341330127, reg_lambda_v=0.366821811291432,
    #           subsample_v=0.879353494846134, date=4, target_cut_values=0.3, is_zhangting=True)
    # print(day + " done")
    #
    # day = "5day_0.3"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=0.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=3, min_child_weight_v=3.402676724513391,
    #           estimators_v=10000, reg_alpha_v=0.587806341330127, reg_lambda_v=0.366821811291432,
    #           subsample_v=0.879353494846134, date=5, target_cut_values=0.3, is_zhangting=True)
    # print(day + " done")
    #
    # day = "30day_100"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=30, target_cut_values=1, is_zhangting=True)
    # print(day + " done")
    #
    # day = "10day_0.1"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=10, target_cut_values=0.1, is_zhangting=False)
    # print(day + " done")
    #
    # day = "9day_0.09"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=9, target_cut_values=0.09, is_zhangting=False)
    # print(day + " done")
    #
    # day = "8day_0.08"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=8, target_cut_values=0.08, is_zhangting=False)
    # print(day + " done")
    #
    # day = "7day_0.07"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=7, target_cut_values=0.07, is_zhangting=False)
    # print(day + " done")
    #
    # day = "6day_0.06"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=6, target_cut_values=0.006, is_zhangting=False)
    # print(day + " done")
    #
    # day = "5day_0.05"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=5, target_cut_values=0.0, is_zhangting=False)
    # print(day + " done")
    #
    # day = "4day_0.04"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=4, target_cut_values=0.04, is_zhangting=False)
    # print(day + " done")
    #
    # day = "3day_0.03"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=3, target_cut_values=0.03, is_zhangting=False)
    # print(day + " done")
    #
    # day = "2day_0.02"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=2, target_cut_values=0.02, is_zhangting=False)
    # print(day + " done")
    #
    # day = "1day_0.02"
    # xgb_train(indicators=indicators, model_name=day + "_model", sample_name='_zhuban_new_indicators_newtarget_',
    #           Dataframe=df, target="target", colsample_bytree_v=0.8834044009405149, gamma_v=3.406489868843162,
    #           learning_rate_v=0.00030113231069171437, max_depth_v=5, min_child_weight_v=10.402676724513391,
    #           estimators_v=10000, reg_alpha_v=3.587806341330127, reg_lambda_v=1.366821811291432,
    #           subsample_v=0.879353494846134, date=1, target_cut_values=0.02, is_zhangting=False)
    # print(day + " done")
    # # day = "10day_neg"
