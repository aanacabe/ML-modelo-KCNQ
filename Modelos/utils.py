import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def Data_Preprocess(dataKCNQ1,dataKCNQ2,dataKCNQ3,dataKCNQ4,dataKCNQ5):
    # Añade las variables que le faltan a los datos de cada gen, añade una variable para identificar a que gen pertenece cada dato (mediante one-hot encoding) y escala las variables continuas mediante Z-score normalization
    # Args:
    #     dataKCNQi (pd.DataFrame): datos correspondientes al gen KCNQi
    # 
    # Returns:
    #     Tuple[lista de pd.DataFrame, lista de pd.Series, pd.Index]: lista de 5 dataframes con features (uno por cada gen), lista de 5 series con la variable objetivo y index con nombres de las features

    
    ##Primero se añaden las variables que indican a que gen corresponde cada dato, y se añaden las variables que le faltan a los datos de cada gen
    
    # dataKCNQ1['Channel']=1
    dataKCNQ1['KCNQ1']=1
    dataKCNQ1['KCNQ2']=0
    dataKCNQ1['KCNQ3']=0
    dataKCNQ1['KCNQ4']=0
    dataKCNQ1['KCNQ5']=0

    #dataKCNQ2['Channel']=2
    dataKCNQ2['KCNQ1']=0
    dataKCNQ2['KCNQ2']=1
    dataKCNQ2['KCNQ3']=0
    dataKCNQ2['KCNQ4']=0
    dataKCNQ2['KCNQ5']=0
    
    idx=dataKCNQ2.columns.get_loc('coil')
    dataKCNQ2.insert(idx+1,'beta_strand',0)

    # dataKCNQ3['Channel']=3
    dataKCNQ3['KCNQ1']=0
    dataKCNQ3['KCNQ2']=0
    dataKCNQ3['KCNQ3']=1
    dataKCNQ3['KCNQ4']=0
    dataKCNQ3['KCNQ5']=0
    
    idx=dataKCNQ3.columns.get_loc('Initial_A')
    dataKCNQ3.insert(idx+1,'Initial_C',0)

    # dataKCNQ4['Channel']=4
    dataKCNQ4['KCNQ1']=0
    dataKCNQ4['KCNQ2']=0
    dataKCNQ4['KCNQ3']=0
    dataKCNQ4['KCNQ4']=1
    dataKCNQ4['KCNQ5']=0
    
    idx=dataKCNQ4.columns.get_loc('Initial_A')
    dataKCNQ4.insert(idx+1,'Initial_C',0)
    idx=dataKCNQ4.columns.get_loc('coil')
    dataKCNQ4.insert(idx+1,'beta_strand',0)
    

    # dataKCNQ5['Channel']=5
    dataKCNQ5['KCNQ1']=0
    dataKCNQ5['KCNQ2']=0
    dataKCNQ5['KCNQ3']=0
    dataKCNQ5['KCNQ4']=0
    dataKCNQ5['KCNQ5']=1
    
    idx=dataKCNQ5.columns.get_loc('p_to_p')
    dataKCNQ5.insert(idx+1,'a_to_a',0)
    
    data=pd.concat([dataKCNQ1,dataKCNQ2,dataKCNQ3,dataKCNQ4,dataKCNQ5],ignore_index=True)

    ##Se separan las features de la variable objetivo

    X1=dataKCNQ1.drop(['Variation','My_Label'], axis=1)
    y1=dataKCNQ1.My_Label
    
    X2=dataKCNQ2.drop(['Variation','My_Label'], axis=1)
    y2=dataKCNQ2.My_Label
    
    X3=dataKCNQ3.drop(['Variation','My_Label'], axis=1)
    y3=dataKCNQ3.My_Label
    
    X4=dataKCNQ4.drop(['Variation','My_Label'], axis=1)
    y4=dataKCNQ4.My_Label
    
    X5=dataKCNQ5.drop(['Variation','My_Label'], axis=1)
    y5=dataKCNQ5.My_Label

    ## Por último se escalan las variables continuas

    categorical = []
    continous = []
    for column in data.columns:
        if len(data[column].unique()) < 10:
            categorical.append(column)
        else:
            continous.append(column)
    continous.remove('Variation')

    scaler= StandardScaler()
    scaler.fit(data[continous])
    X1[continous]=scaler.transform(X1[continous])
    X2[continous]=scaler.transform(X2[continous])
    X3[continous]=scaler.transform(X3[continous])
    X4[continous]=scaler.transform(X4[continous])
    X5[continous]=scaler.transform(X5[continous])

    X=[X1,X2,X3,X4,X5]
    y=[y1,y2,y3,y4,y5]

    features=X1.columns
      
    return X,y,features


def KCNQ_Kfold(X,y,kfold):
    # Aplica StratifiedKFold en los datos de cada gen por separado y luego los une, cada fold final tiene la misma proporción de datos de cada gen que el dataset original
    # 
    # Args:
    #     X (lista de 5 pd.DataFrame): lista con las features de cada gen KCNQ
    #     y (lista de 5 pd.Series): lista con la variable objetivo de cada gen KCNQ
    #     kfold (int): Número de folds
    # 
    # Returns:
    #     Tuple[lista de listas, lista de listas]: lista de kfold listas con las features de cada fold, lista de kfold listas con la variable objetivo de cada fold
   
    skf=StratifiedKFold(n_splits=kfold)
    X1_cv=[]
    y1_cv=[]
    for i,(train_index, test_index) in enumerate(skf.split(X[0],y[0])):
        X1_cv.append(X[0].to_numpy()[test_index])
        y1_cv.append(y[0].to_numpy()[test_index])
    X2_cv=[]
    y2_cv=[]
    for i,(train_index, test_index) in enumerate(skf.split(X[1],y[1])):
        X2_cv.append(X[1].to_numpy()[test_index])
        y2_cv.append(y[1].to_numpy()[test_index])
    X3_cv=[]
    y3_cv=[]
    for i,(train_index, test_index) in enumerate(skf.split(X[2],y[2])):
        X3_cv.append(X[2].to_numpy()[test_index])
        y3_cv.append(y[2].to_numpy()[test_index])
    X4_cv=[]
    y4_cv=[]
    for i,(train_index, test_index) in enumerate(skf.split(X[3],y[3])):
        X4_cv.append(X[3].to_numpy()[test_index])
        y4_cv.append(y[3].to_numpy()[test_index])
    X5_cv=[]
    y5_cv=[]
    for i,(train_index, test_index) in enumerate(skf.split(X[4],y[4])):
        X5_cv.append(X[4].to_numpy()[test_index])
        y5_cv.append(y[4].to_numpy()[test_index])
    X_cv=[]
    y_cv=[]
    for i in range(kfold):
        X_cv.append(np.concatenate([X1_cv[i],X2_cv[i],X3_cv[i],X4_cv[i],X5_cv[i]]).tolist())
        y_cv.append(np.concatenate([y1_cv[i],y2_cv[i],y3_cv[i],y4_cv[i],y5_cv[i]]).tolist())
    #X_cv and y_cv are lists of numpy arrays

    return X_cv, y_cv

def sensitivity(model, X, y):
    # Calcula el valor de la sensibilidad

    # Args:
    #     model: Modelo entrenado
    #     X (pd.DataFrame) : DataFrame con las features
    #     y (pd.Series) : Serie con la variable objetivo
    # Returns: 
    #     float : Valor de la sensibilidad
    
    cm=confusion_matrix(y, model.predict(X))
    sensitivity=cm[1,1]/(cm[1,1]+cm[1,0])
    return sensitivity   
def specificity(model, X, y):
    # Calcula el valor de la especificidad

    # Args:
    #     model: Modelo entrenado
    #     X (pd.DataFrame) : DataFrame con las features
    #     y (pd.Series) : Serie con la variable objetivo
    # Returns: 
    #     float : Valor de la especificidad
    
    cm=confusion_matrix(y, model.predict(X))
    specificity=cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity
def evaluation_kfold(model, X, y,ite, kfold,sens,spec,roc,features):
    # Toma arrays de dimensiones (6,kfold) con las métricas del modelo y los devuelve habiendoles añadido los valores correspondientes a la iteración actual
    # Args:
    #     model: Modelo entrenado, debe tener el atributo predict_proba
    #     X (pd.DataFrame) : DataFrame con las features
    #     y (pd.Series) : Serie con la variable objetivo
    #     ite (int) : iteración del proceso de validación cruzada al que corresponden el modelo,X e y
    #     kfold (int) : Número de folds del proceso de validación cruzada (número total de iteraciones)
    #     sens (np.array) : array de dimensiones (6,kfold) para guardar en cada iteración los valores de sensibilidad del dataset total y de cada gen KCNQ
    #     spec (np.array) : array de dimensiones (6,kfold) para guardar en cada iteración los valores de especificidad del dataset total y de cada gen KCNQ
    #     roc (np.array : array de dimensiones (6,kfold) para guardar en cada iteración los valores de AUC_ROC del dataset total y de cada gen KCNQ
    #     features (pd.Index) : Nombres de las features con las que ha sido entrenado el modelo
    # Returns:
    #     Tuple[np.array,np.array,np.array] : arrays de dimensiones (6,kfold) conteniendo los valores de sensibilidad, especificidad y AUC-ROC en ese orden
    
    sens[0,ite]=sensitivity(model,X[features],y)
    spec[0,ite]=specificity(model,X[features],y)
    roc[0,ite]=roc_auc_score(y,model.predict_proba(X[features])[:,1])
    X1=X[X['KCNQ1']==1]
    y1=y[X['KCNQ1']==1]
    sens[1,ite]=sensitivity(model,X1[features],y1)
    spec[1,ite]=specificity(model,X1[features],y1)
    roc[1,ite]=roc_auc_score(y1,model.predict_proba(X1[features])[:,1])
    X2=X[X['KCNQ2']==1]
    y2=y[X['KCNQ2']==1]
    sens[2,ite]=sensitivity(model,X2[features],y2)
    spec[2,ite]=specificity(model,X2[features],y2)
    roc[2,ite]=roc_auc_score(y2,model.predict_proba(X2[features])[:,1])
    X3=X[X['KCNQ3']==1]
    y3=y[X['KCNQ3']==1]
    sens[3,ite]=sensitivity(model,X3[features],y3)
    spec[3,ite]=specificity(model,X3[features],y3)
    roc[3,ite]=roc_auc_score(y3,model.predict_proba(X3[features])[:,1])
    X4=X[X['KCNQ4']==1]
    y4=y[X['KCNQ4']==1]
    sens[4,ite]=sensitivity(model,X4[features],y4)
    spec[4,ite]=specificity(model,X4[features],y4)
    roc[4,ite]=roc_auc_score(y4,model.predict_proba(X4[features])[:,1])
    X5=X[X['KCNQ5']==1]
    y5=y[X['KCNQ5']==1]
    sens[5,ite]=sensitivity(model,X5[features],y5)
    spec[5,ite]=specificity(model,X5[features],y5)
    roc[5,ite]=roc_auc_score(y5,model.predict_proba(X5[features])[:,1])

    return sens,spec,roc
