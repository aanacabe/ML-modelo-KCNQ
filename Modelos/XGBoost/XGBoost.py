import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
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

    ##Selección de features
    
    xgb_model = XGBClassifier()
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=xgb_model, min_features_to_select=1, cv=cv, scoring='roc_auc',n_jobs=-1)
    selector.fit(X_train, y_train)

    if i==0:
        ## Gráfico de la selección de variables correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(selector.cv_results_)
        plt.figure()
        plt.xlabel("Número de variables")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        plt.plot(cv_results["n_features"],cv_results["mean_test_score"],marker='o',color='tab:blue')
        plt.title("RFE para XGBoost")
        plt.savefig('RFEXGB.png')

    print(f"Optimal number of features: {selector.n_features_}")
    
    features_xgb=list(selector.get_feature_names_out())

    ##Ajuste de hiperparámetros
    
    xgb_model = XGBClassifier(n_estimators=500,max_depth=1)
    params={
        'gamma':np.linspace(0,1,10),
        'learning_rate':np.logspace(-2,0,10)
    }
    
    xgb_cv = GridSearchCV(xgb_model, params,scoring="roc_auc", cv=cv, verbose=1,n_jobs=-1)
    xgb_cv.fit(X_train[features_xgb], y_train)
    best_params1 = xgb_cv.best_params_
    print(best_params1)

    if i==0:
        ## Gráfico del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(xgb_cv.cv_results_)
        Gamma=np.linspace(0,1,10)
        Learning_Rate=np.logspace(-2,0,10)
        Z=np.zeros((len(Gamma),len(Learning_Rate)))
        for i1 in range(len(Gamma)):
            for i2 in range(len(Learning_Rate)):
                Z[i1,i2]=cv_results[(cv_results['param_gamma']==Gamma[i1])&(cv_results['param_learning_rate']==Learning_Rate[i2])]['mean_test_score']
        
        plt.figure()
        plt.xlabel("Learning Rate")
        plt.ylabel("Gamma")
        plt.grid(True,color='k',linestyle=':')
        contour=plt.contourf(Learning_Rate, Gamma, Z)
        plt.colorbar(contour, label='ROC-AUC')
        plt.xscale('log')
        plt.title("Ajuste de Hiperparámetros para XGBoost")
        plt.savefig('HypTunXGB1.png')

    
    xgb_model = XGBClassifier(n_estimators=500,max_depth=1,**best_params1)
    params={
        'subsample':np.linspace(0.6,1,10),
        'colsample_bytree':np.linspace(0.2,0.9,10),
    }
    
    xgb_cv = GridSearchCV(xgb_model, params,scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    xgb_cv.fit(X_train[features_xgb], y_train)
    best_params2 = xgb_cv.best_params_
    print(best_params2)

    if i==0:
        ## Gráfico del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(xgb_cv.cv_results_)
        Subsample=np.linspace(0.6,1,10)
        Colsample=np.linspace(0.2,0.9,10)
        Z=np.zeros((len(Subsample),len(Colsample)))
        for i1 in range(len(Subsample)):
            for i2 in range(len(Colsample)):
                Z[i1,i2]=cv_results[(cv_results['param_subsample']==Subsample[i1])&(cv_results['param_colsample_bytree']==Colsample[i2])]['mean_test_score']
        
        plt.figure()
        plt.ylabel("Subsample")
        plt.xlabel("Colsample_bytree")
        plt.grid(True,color='k',linestyle=':')
        contour=plt.contourf(Colsample, Subsample, Z)
        plt.colorbar(contour, label='ROC-AUC')
        plt.title("Ajuste de Hiperparámetros para XGBoost")
        plt.savefig('HypTunXGB2.png')

    
    xgb_model = XGBClassifier(n_estimators=500,max_depth=1,**best_params1,**best_params2)
    xgb_model.fit(X_train[features_xgb], y_train)
    sens_xgb,spec_xgb,roc_xgb=evaluation_kfold(xgb_model, X_test, y_test,i, kfold,sens_xgb,spec_xgb,roc_xgb,features_xgb)
    
scores_xgb={('sensitivity','mean'):np.mean(sens_xgb,axis=1),
           ('sensitivity','std'):np.std(sens_xgb,axis=1),
           ('specificity','mean'):np.mean(spec_xgb,axis=1),
           ('specificity','std'):np.std(spec_xgb,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_xgb,axis=1),
           ('AUC-ROC','std'):np.std(roc_xgb,axis=1),}
df_scores_xgb=pd.DataFrame(scores_xgb, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_xgb.to_csv('scores_xgb.csv')