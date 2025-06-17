import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from utils import *

dataKCNQ1=pd.read_excel('../../data/KCNQ1/KCNQ1_encoded.xlsx')
dataKCNQ2=pd.read_excel('../../data/KCNQ2/KCNQ2_encoded.xlsx')
dataKCNQ3=pd.read_excel('../../data/KCNQ3/KCNQ3_encoded.xlsx')
dataKCNQ4=pd.read_excel('../../data/KCNQ4/KCNQ4_encoded.xlsx')
dataKCNQ5=pd.read_excel('../../data/KCNQ5/KCNQ5_encoded.xlsx')

kfold=5
#Prepocesamiento de los datos
X,y,features=Data_Preprocess(dataKCNQ1,dataKCNQ2,dataKCNQ3,dataKCNQ4,dataKCNQ5)
#Separar los datos en subconjuntos para utilizar validación cruzada
X_cv,y_cv=KCNQ_Kfold(X,y,kfold)

def select_columns(X, columns): #Para poder utilizar features distintas para entrenar cada modelo del clasificador
    return X[columns]

sens_lr=np.zeros((6,kfold)) #6 filas, la primera para todo el dataset, y del resto de filas una para cada gen KCNQ
spec_lr=np.zeros((6,kfold)) #Kfold columnas, en cada una se guarda el rendimiento de una iteración del método de validación cruzada
roc_lr=np.zeros((6,kfold))
for i in range(kfold):
    X_test=pd.DataFrame(X_cv[i],columns=features)
    y_test=pd.Series(y_cv[i])
    X_train=[]
    y_train=[]
    for j in range(kfold):
        if i!=j:
            X_train.extend(X_cv[j])
            y_train.extend(y_cv[j])
    X_train=pd.DataFrame(X_train,columns=features)
    y_train=pd.Series(y_train)

    ## LOGISTIC REGRESSION
    log_model = LogisticRegression(random_state=1)
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=log_model, min_features_to_select=1, cv=cv, scoring='roc_auc')
    selector.fit(X_train, y_train)

    features_lr=list(selector.get_feature_names_out())
    params={
        "C": np.logspace(-1,1,30),
        "solver":["liblinear","lbfgs","newton-cholesky"]
    }
    log_cv = GridSearchCV(log_model, params, scoring="roc_auc", cv=cv, verbose=0)
    log_cv.fit(X_train[features_lr], y_train)
    best_params_lr = log_cv.best_params_
    
    log_model = LogisticRegression(**best_params_lr)

    ## SVM

    svc_model = SVC(degree=2)
    sfs1 = SFS(svc_model, k_features='best', forward=False, floating=False, verbose=0,scoring='roc_auc',n_jobs=-1,cv=cv)
    sfs1 = sfs1.fit(X_train, y_train)

    features_svc=list(sfs1.k_feature_names_)
    
    params={
        'C': np.logspace(-2,1,20),
        'gamma': np.logspace(-3,0,20),
        'kernel': ['linear','poly','rbf']
    }
    
    svc_cv=GridSearchCV(svc_model, params, scoring='roc_auc', cv=cv, verbose=0,n_jobs=-1)
    svc_cv.fit(X_train[features_svc], y_train)
    best_params_svc=svc_cv.best_params_

    svc_model = SVC(**best_params_svc,probability=True,degree=2)

    ## RANDOM FOREST

    rforest_model = RandomForestClassifier(criterion='entropy',random_state=1)
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=rforest_model, min_features_to_select=1, cv=cv, scoring='roc_auc',n_jobs=-1)
    selector.fit(X_train, y_train)

    features_rf=list(selector.get_feature_names_out())
    params = {
        "n_estimators" : [1000],
        "criterion":['gini','entropy'],
        "max_depth" : [6,8,10,12],
        "min_samples_split":[2,4,6]
    }
    rforest_cv = GridSearchCV(rforest_model, params, scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    rforest_cv.fit(X_train[features_rf],y_train)
    best_params_rf = rforest_cv.best_params_

    rforest_model = RandomForestClassifier(**best_params_rf)

    ## XGBOOST

    xgb_model = XGBClassifier()
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=xgb_model, min_features_to_select=1, cv=cv, scoring='roc_auc',n_jobs=-1)
    selector.fit(X_train, y_train)

    features_xgb=list(selector.get_feature_names_out())
    xgb_model = XGBClassifier(n_estimators=500,max_depth=1)
    params={
        'gamma':np.linspace(0,1,10),
        'learning_rate':np.logspace(-2,0,10)
    }
    
    xgb_cv = GridSearchCV(xgb_model, params,scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    xgb_cv.fit(X_train[features_xgb], y_train)
    best_params1_xgb = xgb_cv.best_params_

    xgb_model = XGBClassifier(n_estimators=500,max_depth=1,**best_params1_xgb)
    params={
        'subsample':np.linspace(0.6,1,10),
        'colsample_bytree':np.linspace(0.2,0.9,10),
    }
    
    xgb_cv = GridSearchCV(xgb_model, params,scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    xgb_cv.fit(X_train[features_xgb], y_train)
    best_params2_xgb = xgb_cv.best_params_

    xgb_model = XGBClassifier(n_estimators=500,max_depth=1,**best_params1_xgb,**best_params2_xgb)

    log_pipe = Pipeline([
        ('selector', FunctionTransformer(select_columns, kw_args={'columns': features_lr}, validate=False)),
        ('classifier', log_model)
    ])
    
    svc_pipe= Pipeline([
        ('selector', FunctionTransformer(select_columns, kw_args={'columns': features_svc}, validate=False)),
        ('classifier', svc_model)
    ])
    
    rforest_pipe = Pipeline([
        ('selector', FunctionTransformer(select_columns, kw_args={'columns': features_rf}, validate=False)),
        ('classifier', rforest_model)
    ])

    xgb_pipe= Pipeline([
        ('selector', FunctionTransformer(select_columns, kw_args={'columns': features_xgb}, validate=False)),
        ('classifier', xgb_model)
    ])
    
    estimators=[('lr', log_pipe), ('svc', svc_pipe), ('rf', rforest_pipe),('xgb',xgb_pipe)]
    soft_vote = VotingClassifier(estimators, voting='soft')
    soft_vote.fit(X_train, y_train)
    
    sens_soft,spec_soft,roc_soft=evaluation_kfold(soft_vote, X_test, y_test, i, kfold,sens_soft,spec_soft,roc_soft,X_train.columns)
    
scores_soft={('sensitivity','mean'):np.mean(sens_soft,axis=1),
           ('sensitivity','std'):np.std(sens_soft,axis=1),
           ('specificity','mean'):np.mean(spec_soft,axis=1),
           ('specificity','std'):np.std(spec_soft,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_soft,axis=1),
           ('AUC-ROC','std'):np.std(roc_soft,axis=1),}
df_scores_soft=pd.DataFrame(scores_soft, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_soft.to_csv('scores_soft.csv')