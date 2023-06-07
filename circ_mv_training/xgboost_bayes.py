
from logging.config import valid_ident
import os 
import pickle
import xgboost as xgb
import numpy as np
import scipy
import sklearn
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.feature_selection import RFECV, RFE
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from functools import partial
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_validate
print("scipy version :", scipy.__version__)

import pickle
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import RFECV, RFE
from functools import partial
import json
import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
# import ZZTime as zz
# zzTime = zz.outputTime()



def train_bayes(target, Dataframe, indicators, sample_name):
    rng = np.random.RandomState(31337)

    ##############
    ## options ##
    ##############
    FeatureSelection = False
    RFESelection = False
    n_features = 10
    GridSearch = False
    print("\n sample is :", sample_name)
    print("\n training target is :", target)
    print("indicators:" ,indicators)
    data = Dataframe
    traindataset, valdataset  = train_test_split(data, test_size=0.3, random_state=rng)
    nS = len(traindataset.iloc[(getattr(traindataset,target).values == 1)])
    nB = len(traindataset.iloc[(getattr(traindataset,target).values == 0)])
    print("nB/nS:",nB/nS)
    pbounds = {
        'learning_rate': (0.01, 1.0),
        'n_estimators': (100, 10000),
        'max_depth': (3,20),
        'subsample': (0.8, 1.0),  # Change for big datasets
        'colsample_bytree': (0.8, 1.0),  # Change for datasets with lots of features
        'reg_alpha': (0, 10),  
        'reg_lambda': (0, 10), 
        'gamma': (0, 10),
        'min_child_weight':(0,10)}
    def xgboost_hyper_param(learning_rate,
                            n_estimators,
                            max_depth,
                            subsample,
                            colsample_bytree,
                            reg_alpha,
                            reg_lambda,
                            gamma,
                            min_child_weight
                            ):

        max_depth = int(max_depth)
        n_estimators = int(n_estimators)

        model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            gamma=gamma,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            scale_pos_weight=nB/nS,
            tree_method='gpu_hist'
            )

        model.fit(
            traindataset[indicators].values,
            getattr(traindataset,target).values
            )       
        probaT = model.predict_proba(valdataset[indicators].values.astype(np.float64))
        fprt, tprt, thresholds = roc_curve(valdataset[target].astype(np.float64), probaT[:,1])
        test_auct = auc(np.sort(fprt), np.sort(tprt))
        proba = model.predict_proba(traindataset[indicators].values)
        fpr, tpr, thresholds = roc_curve(traindataset[target].values,proba[:,1])
        train_auc = auc(np.sort(fpr), np.sort(tpr))
        print("ZZ: test_auct = ",test_auct)
        print("ZZ : reg_alpha = ",reg_alpha)
        return test_auct
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=5,n_iter=100)

    #Extracting the best parameters
    params = optimizer.max['params']
    print(params)


if __name__ == "__main__":

    df = pd.read_csv("/pycharm/finance_spider_data_analysis-main/FT_XGBoost-develop/csv/xgboost_train_zhuban.csv")
    indicators = ['T3','EMA','HT_TRENDLINE','KAMA','SAR','ADX','APO','AROONOSC','CCI','CMO','DX','macd','macd_signal','macd_hist','fix_macd','fix_macd_signal','fix_macd_hist','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DI','PLUS_DM','PPO','ROC','ROCP','ROCR','ROCR100','RSI','slowk','slowd','slowj','fastk','fastd','TRIX','ULTOSC','WILLR','AD','ADOSC','HT_DCPERIOD','HT_DCPHASE','inphase','quadrature','sine','leadsine','HT_TRENDMODE','aroondown','aroonup','arroon_dif','bop','ATR','NATR','TRANGE','vol_r','mfi','MOM','buy_sm_vol','buy_md_vol','buy_lg_vol','buy_elg_vol','sell_sm_vol','sell_md_vol','sell_lg_vol','sell_elg_vol','MTM']
    train_bayes(target="target_1day_chg10", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_1day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_2day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_3day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_4day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_5day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_6day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_7day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_8day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_9day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_10day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_20day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_30day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_40day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_50day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    train_bayes(target="target_60day", Dataframe= df, indicators=indicators,sample_name="zhuban")
    
    