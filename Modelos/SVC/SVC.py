import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.svm import SVC
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
    
    cv = StratifiedKFold(5)
    svc_model = SVC(degree=2)
    sfs1 = SFS(svc_model, k_features='best', forward=False, floating=False, verbose=0,scoring='roc_auc',n_jobs=-1,cv=cv)
    sfs1 = sfs1.fit(X_train, y_train)
    dictio=sfs1.subsets_
    score=[]
    if i==0:
        ## Gráfico de la selección de variables correspondiente a la primera iteración de validación cruzada
        for j in range(1,X_train.shape[1]+1):
            score.append(dictio[j]['avg_score'])
        plt.figure()
        plt.xlabel("Número de variables")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        plt.plot(np.arange(1,X_train.shape[1]+1),score,marker='o',color='tab:blue')
        plt.title("SBS para SVM")
        plt.savefig('SBSSVM.png')

    print(f"Optimal number of features: {len(sfs1.k_feature_names_)}")
    features_svc=list(sfs1.k_feature_names_)

    ##Ajuste de hiperparámetros
    
    params={
        'C': np.logspace(-2,1,20),
        'gamma': np.logspace(-3,0,20),
        'kernel': ['linear','poly','rbf']
    }
    
    svc_cv=GridSearchCV(svc_model, params, scoring='roc_auc', cv=cv, verbose=0,n_jobs=-1)
    svc_cv.fit(X_train[features_svc], y_train)
    best_params=svc_cv.best_params_

    if i==0:
        ## Gráficos del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(svc_cv.cv_results_)
        linear=cv_results[cv_results['param_kernel']=='linear']
        poly=cv_results[cv_results['param_kernel']=='poly']
        rbf=cv_results[cv_results['param_kernel']=='rbf']
        Gamma=np.logspace(-3,0,20)
        C=np.logspace(-2,1,20)
        linear_Z=np.zeros((len(Gamma),len(C)))
        for i1 in range(len(Gamma)):
            for i2 in range(len(C)):
                linear_Z[i1,i2]=linear[(linear['param_gamma']==Gamma[i1])&(linear['param_C']==C[i2])]['mean_test_score']
        
        plt.figure()
        plt.xlabel("C")
        plt.ylabel("Gamma")
        plt.grid(True,color='k',linestyle=':')
        contour=plt.contourf(C, Gamma, linear_Z)
        plt.colorbar(contour, label='ROC-AUC')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Ajuste de Hiperparámetros con kernel lineal")
        plt.savefig('HypTunSVMLin.png')

        poly_Z=np.zeros((len(Gamma),len(C)))
        for i1 in range(len(Gamma)):
            for i2 in range(len(C)):
                poly_Z[i1,i2]=poly[(poly['param_gamma']==Gamma[i1])&(poly['param_C']==C[i2])]['mean_test_score']
        
        plt.figure()
        plt.xlabel("C")
        plt.ylabel("Gamma")
        plt.grid(True,color='k',linestyle=':')
        contour=plt.contourf(C, Gamma, poly_Z)
        plt.colorbar(contour, label='ROC-AUC')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Ajuste de Hiperparámetros con kernel polinomial")
        plt.savefig('HypTunSVMPoly.png')

        rbf_Z=np.zeros((len(Gamma),len(C)))
        for i1 in range(len(Gamma)):
            for i2 in range(len(C)):
                rbf_Z[i1,i2]=rbf[(rbf['param_gamma']==Gamma[i1])&(rbf['param_C']==C[i2])]['mean_test_score']
        
        plt.figure()
        plt.xlabel("C")
        plt.ylabel("Gamma")
        plt.grid(True,color='k',linestyle=':')
        contour=plt.contourf(C, Gamma, rbf_Z)
        plt.colorbar(contour, label='ROC-AUC')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Ajuste de Hiperparámetros con kernel radial")
        plt.savefig('HypTunSVMRad.png')        
    print(best_params)
    svc_model = SVC(**best_params,probability=True,degree=2)
    svc_model.fit(X_train[features_svc], y_train)
    sens_svc,spec_svc,roc_svc=evaluation_kfold(svc_model, X_test, y_test,i, kfold,sens_svc,spec_svc,roc_svc,features_svc)
    
scores_svc={('sensitivity','mean'):np.mean(sens_svc,axis=1),
           ('sensitivity','std'):np.std(sens_svc,axis=1),
           ('specificity','mean'):np.mean(spec_svc,axis=1),
           ('specificity','std'):np.std(spec_svc,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_svc,axis=1),
           ('AUC-ROC','std'):np.std(roc_svc,axis=1),}
df_scores_svc=pd.DataFrame(scores_svc, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_svc.to_csv('scores_svc.csv')