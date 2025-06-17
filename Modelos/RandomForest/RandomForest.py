import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
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
    
    rforest_model = RandomForestClassifier(criterion='entropy',random_state=1)
    cv = StratifiedKFold(5)
    selector=RFECV(estimator=rforest_model, min_features_to_select=1, cv=cv, scoring='roc_auc',n_jobs=-1)
    selector.fit(X_train, y_train)

    if i==0:
        ## Gráfico de la selección de variables correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(selector.cv_results_)
        plt.figure()
        plt.xlabel("Número de variables")
        plt.ylabel("ROC-AUC")
        plt.grid(True,color='k',linestyle=':')
        #plt.ylim(0.92,0.96)
        plt.plot(cv_results["n_features"],cv_results["mean_test_score"],marker='o',color='tab:blue')
        # plt.errorbar(
        #     x=cv_results["n_features"],
        #     y=cv_results["mean_test_score"],
        #     yerr=cv_results["std_test_score"],
        # )
        plt.title("RFE para Random Forest")
        plt.savefig('RFERF.png')
        
    print(f"Optimal number of features: {selector.n_features_}")
        
    features_rf=list(selector.get_feature_names_out())

    ##Ajuste de hiperparámetros
    
    params = {
        "n_estimators" : [1000],
        "criterion":['gini','entropy'],
        "max_depth" : [6,8,10,12],
        "min_samples_split":[2,4,6]
    }
    rforest_cv = GridSearchCV(rforest_model, params, scoring="roc_auc", cv=cv, verbose=0,n_jobs=-1)
    rforest_cv.fit(X_train[features_rf],y_train)
    best_params = rforest_cv.best_params_
    if i==0:
        ## Gráficos del ajuste de hiperparámetros correspondiente a la primera iteración de validación cruzada
        cv_results = pd.DataFrame(rforest_cv.cv_results_)
        gini=cv_results[cv_results['param_criterion']=='gini']
        entropy=cv_results[cv_results['param_criterion']=='entropy']
        df_results = pd.DataFrame({
            'param_max_depth': gini['param_max_depth'],
            'param_min_samples_split': gini['param_min_samples_split'],
            'mean_test_score': gini['mean_test_score']
        })
        
        # Pivot the DataFrame to get a 2D array suitable for a heatmap
        # 'index' will be one hyperparameter, 'columns' will be the other,
        # and 'values' will be the mean test score.
        heatmap_data = df_results.pivot(index='param_min_samples_split', columns='param_max_depth', values='mean_test_score')
        
        
        plt.figure(figsize=(10, 8)) # Set the figure size for better readability
        sns.heatmap(
            heatmap_data,
            annot=True,          # Annotate cells with the score values
            cmap='viridis',      # Choose a colormap (e.g., 'viridis', 'YlGnBu', 'RdYlGn')
            fmt=".3f",           # Format annotation values to 3 decimal places
            linewidths=.5       # Add lines between cells for better separation
        )
        
        plt.title('Rendimiento de Random Forest con Criterio Gini', fontsize=16)
        plt.xlabel('Profundidad Máxima', fontsize=14)
        plt.ylabel('Mínimo de ejemplos', fontsize=14)
        plt.xticks(rotation=45) # Rotate x-axis labels if they overlap
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.tight_layout()      # Adjust layout to prevent labels from overlapping
        plt.savefig('HypTunRFGini.png')


        df_results = pd.DataFrame({
            'param_max_depth': entropy['param_max_depth'],
            'param_min_samples_split': entropy['param_min_samples_split'],
            'mean_test_score': entropy['mean_test_score']
        })
        
        # Pivot the DataFrame to get a 2D array suitable for a heatmap
        # 'index' will be one hyperparameter, 'columns' will be the other,
        # and 'values' will be the mean test score.
        heatmap_data = df_results.pivot(index='param_min_samples_split', columns='param_max_depth', values='mean_test_score')
        
        
        plt.figure(figsize=(10, 8)) # Set the figure size for better readability
        sns.heatmap(
            heatmap_data,
            annot=True,          # Annotate cells with the score values
            cmap='viridis',      # Choose a colormap (e.g., 'viridis', 'YlGnBu', 'RdYlGn')
            fmt=".3f",           # Format annotation values to 3 decimal places
            linewidths=.5       # Add lines between cells for better separation
        )
        
        plt.title('Rendimiento de Random Forest con Criterio Entropy', fontsize=16)
        plt.xlabel('Profundidad Máxima', fontsize=14)
        plt.ylabel('Mínimo de ejemplos', fontsize=14)
        plt.xticks(rotation=45) # Rotate x-axis labels if they overlap
        plt.yticks(rotation=0)  # Keep y-axis labels horizontal
        plt.tight_layout()      # Adjust layout to prevent labels from overlapping
        plt.savefig('HypTunRFEntropy.png')

    print(best_params)
    rforest_model = RandomForestClassifier(**best_params)
    rforest_model.fit(X_train[features_rf], y_train)
    sens_rf,spec_rf,roc_rf=evaluation_kfold(rforest_model, X_test, y_test,i, kfold,sens_rf,spec_rf,roc_rf,features_rf)
    
scores_rf={('sensitivity','mean'):np.mean(sens_rf,axis=1),
           ('sensitivity','std'):np.std(sens_rf,axis=1),
           ('specificity','mean'):np.mean(spec_rf,axis=1),
           ('specificity','std'):np.std(spec_rf,axis=1),
           ('AUC-ROC','mean'):np.mean(roc_rf,axis=1),
           ('AUC-ROC','std'):np.std(roc_rf,axis=1),}
df_scores_rf=pd.DataFrame(scores_rf, index=["Total", "KCNQ1", "KCNQ2","KCNQ3","KCNQ4","KCNQ5"])
df_scores_rf.to_csv('scores_rf.csv')