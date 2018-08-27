# -*- coding: utf-8 -*-
"""
@brief : 配置文件，主要用于配置机器学习模型使用哪种特征和机器学习算法
@to use : 修改features_path用于配置使用哪种特征；修改clf_name用于配置使用哪种学习算法；可在clfs_dict中对学习算法的超参数进行修改；
@author: Jian
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
status_vali = True
"""修改features_path"""
features_path = '../features/data_ensemble.pkl'

"""修改clf_name, clfs_dict"""
clf_name = 'xgb'
base_clf = LinearSVC()

clfs = {
    'lg': LogisticRegression(),
    'svm': LinearSVC(),
    'bagging': BaggingClassifier(base_estimator=base_clf, n_estimators=60, max_samples=1.0, max_features=1.0, random_state=1,
                        n_jobs=1, verbose=1),
    'rf': RandomForestClassifier(n_estimators=10, criterion='gini'),
    'adaboost': AdaBoostClassifier(base_estimator=base_clf, n_estimators=50),
    'gbdt': GradientBoostingClassifier(),
    'xgb': xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100, silent=True, objective='multi:softmax',
                        nthread=1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                        colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
                        missing=None)
}
clf = clfs[clf_name]
