import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
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
    
    log_model = LogisticRegression(random_state=1)
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=log_model, min_features_to_select=1, cv=cv, scoring='roc_auc')
    selector.fit(X_train, y_train)
    features_lr=list(selector.get_feature_names_out())

    if i==0:
        ## Gráfico de la selección de variables correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(selector.cv_results_)
        plt.figure()
        plt.xlabel("Número de variables")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        plt.plot(cv_results["n_features"],cv_results["mean_test_score"],marker='o',color='tab:blue')
        plt.title("RFE para Regresión Logística")
        plt.savefig('RFELogReg.png')
        
    print(f"Optimal number of features: {selector.n_features_}")
    
    ##Ajuste de hiperparámetros
    params={
        "C": np.logspace(-1,1,30),
        "solver":["liblinear","lbfgs","newton-cholesky"]
    }
    log_cv = GridSearchCV(log_model, params, scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    log_cv.fit(X_train[features_lr], y_train)
    best_params = log_cv.best_params_
    print(best_params)
    if i==0:
        ## Gráfico del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(log_cv.cv_results_)
        liblinear=cv_results[cv_results['param_solver']=='liblinear']
        lbfgs=cv_results[cv_results['param_solver']=='lbfgs']
        newton_cholesky=cv_results[cv_results['param_solver']=='newton-cholesky']
        plt.figure()
        plt.xlabel("C")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        plt.semilogx(liblinear["param_C"],liblinear["mean_test_score"],marker='o',color='tab:blue',label='liblinear')
        plt.semilogx(lbfgs["param_C"],lbfgs["mean_test_score"],marker='s',color='tab:red',label='lbfgs')
        plt.semilogx(newton_cholesky["param_C"],newton_cholesky["mean_test_score"],marker='^',color='tab:green',label='newton-cholesky')
        plt.title("Ajuste de Hiperparámetros para Regresión Logística")
        plt.legend(loc="upper right")
        plt.savefig('HypTunLogReg.png')
    
    log_model = LogisticRegression(**best_params)
    log_model.fit(X_train[features_lr], y_train)

    sens_lr,spec_lr,roc_lr=evaluation_kfold(log_model, X_test, y_test,i, kfold,sens_lr,spec_lr,roc_lr,features_lr)
    
scores_lr={('sensitivity','mean'):np.mean(sens_lr,axis=1),
           ('sensitivity','std'):np.std(sens_lr,axis=1),
           ('specificity','mean'):np.mean(spec_lr,axis=1),
           ('specificity','std'):np.std(spec_lr,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_lr,axis=1),
           ('AUC-ROC','std'):np.std(roc_lr,axis=1),}
df_scores_lr=pd.DataFrame(scores_lr, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_lr.to_csv('scores_lr.csv')