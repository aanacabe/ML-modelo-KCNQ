import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
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

    ##Ajuste de hiperparámetros

    cv = StratifiedKFold(5)
    knn_model=KNeighborsClassifier()
    params={
        'n_neighbors': np.arange(1,31)
    }
    knn_cv=GridSearchCV(knn_model, params, scoring='roc_auc', cv=cv, verbose=0,n_jobs=-1)
    knn_cv.fit(X_train, y_train)
    best_params = knn_cv.best_params_
    print(best_params)
    if i==0:
        ## Gráficos del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(knn_cv.cv_results_)
        plt.figure()
        plt.xlabel("Número de Vecinos")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        plt.plot(cv_results["param_n_neighbors"],cv_results["mean_test_score"],marker='o',color='tab:blue')
        plt.title("Ajuste de Hiperparámetros para K-Vecinos Próximos")
        plt.savefig('HypTunKNN.png')

    ##Selección de features
    
    knn=KNeighborsClassifier(**best_params)
    sfs1 = SFS(knn, k_features='best', forward=False, floating=False, verbose=0,scoring='roc_auc',n_jobs=-1,cv=cv)
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
        plt.title("SBS para KNN")
        plt.savefig('SBSKNN.png')
        plt.show()
        features_knn_0=list(sfs1.k_feature_names_)
    print(f"Optimal number of features: {len(sfs1.k_feature_names_)}")
    features_knn=list(sfs1.k_feature_names_)
    
    
    knn_model = KNeighborsClassifier(**best_params)
    knn_model.fit(X_train[features_knn], y_train)
    sens_knn,spec_knn,roc_knn=evaluation_kfold(knn_model, X_test, y_test, i, kfold,sens_knn,spec_knn,roc_knn,features_knn)
    
scores_knn={('sensitivity','mean'):np.mean(sens_knn,axis=1),
           ('sensitivity','std'):np.std(sens_knn,axis=1),
           ('specificity','mean'):np.mean(spec_knn,axis=1),
           ('specificity','std'):np.std(spec_knn,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_knn,axis=1),
           ('AUC-ROC','std'):np.std(roc_knn,axis=1),}
df_scores_knn=pd.DataFrame(scores_knn, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_knn.to_csv('scores_knn.csv')