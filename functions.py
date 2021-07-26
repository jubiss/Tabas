import pandas as pd
def merge_data():
    '''
    União dos dados de localização com a base
    '''
    geo_data = pd.read_csv('geo_data.csv')
    geo_data.rename(columns={'id':'location_id'},inplace=True)
    app = pd.read_csv('apartments.csv')
    final_df = app.join(geo_data.set_index('location_id'),on='location_id')
    return final_df

def pre_processing(df):
    '''
    Pré-processamento dos dados para uso no modelo
    '''
    df = df[df['latitude'].isnull()==False]
    df = df[df['bedrooms']!=0]
    df['price_month'] = df['price_month'].str[3:]
    df['price_month'] = df['price_month'].str.replace('.','').astype(int)
    return df

def remove_diff_columns(train,test):
    '''
    Remoção de colunas diferentes em train e test
    '''
    train_column = train.columns.values.tolist()
    test_column = test.columns.values.tolist()
    not_in_train = [i for i in test_column if i not in train_column]
    not_in_test  = [i for i in train_column if i not in test_column]    
    train.drop(not_in_test,axis=1,inplace=True)
    test.drop(not_in_train,axis=1,inplace=True)
    
    return train,test

def bairro(X_train,X_test ,y_train, train_rf=False):
    '''
    Gera as variáveis 'bairro faixa' (Valorização dos bairros) e dummy variables com os bairros com mais dados.
    '''
    X_train['preco_m2'] = y_train/X_train['sqm']

    bairro = X_train.groupby('bairro').mean()
    bairro['bairro faixa'] = pd.qcut(bairro['preco_m2'].sort_values(),10,range(10))
    bairro_dic = bairro['bairro faixa'].to_dict()
    X_train['bairro_faixa'] = X_train['bairro'].map(bairro_dic)
    X_test['bairro_faixa'] = X_test['bairro'].map(bairro_dic)

    X_train['bairro_merged'] = X_train['bairro'].mask(X_train['bairro'].map(X_train['bairro'].value_counts(normalize=True)) < 0.02, 'Other')
    bairro_merged = pd.Series(X_train['bairro_merged'].values,index=X_train['bairro']).to_dict()
    X_test['bairro_merged'] = X_test['bairro'].map(bairro_merged)

    X_train.drop(['preco_m2','bairro'],axis=1,inplace=True)
    X_test.drop(['bairro'],axis=1,inplace=True)

    if train_rf == True:
        X_test['bairro_faixa'].fillna(5,inplace=True)

    X_train = pd.get_dummies(X_train,columns=['bairro_merged'])
    X_test = pd.get_dummies(X_test,columns=['bairro_merged'])
    X_train, X_test = remove_diff_columns(X_train,X_test)
    return X_train,X_test


def find_outliers_turkey(x):
    '''
    Encontra outliers utilizando distancia interquartil
    '''
    import numpy as np
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1
    floor = q1 - 1.5*iqr
    celing = q3 +1.5*iqr
    outlier_indicies = list(x.index[(x<floor)|(x>celing)])

    return outlier_indicies#, outlier_values

def model_validation(df,space_xgb,space_random,train_rf=False,plot = True,com_localizacao=True):
    '''
    Realiza a validação do modelo.
    df -> Dados utilizados,
    space_xgb -> Parametros a serem otimizados no xgboost
    space_random -> Parametros a serem otimizados na Random Forest
    train_rf -> Se deve treinar modelo de Random Forest para servir como Benchmark
    plot -> Plota distribuição de erros.
    com_localizacao -> Defini se variáveis de localização serão adicionadas.
    '''
    import numpy as np
    from skopt import dummy_minimize
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import cross_val_score, KFold
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Parametros
    X = df[['sqm', 'bedrooms','bairro']]
    y = df['price_month']
    n_calls_hyp = 2
    rmse = []
    mae = []
    mape = []
    mae_treino = []
    rmse_treino = []    
    cv_outer = KFold(n_splits=5,shuffle=True)
    param_results = []
    if train_rf==True:
        squared_rf = []
        absolut_rf = []
    #Kfold
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix,:]
        y_train, y_test = y.iloc[train_ix] , y.iloc[test_ix]
    
    #Feature engineering
        if com_localizacao ==True:
            X_train, X_test = bairro(X_train,X_test,y_train,train_rf=train_rf)
        else:
            X_train.drop('bairro',axis=1,inplace=True)
            X_test.drop('bairro',axis=1,inplace=True)
    #Inner Cross-Validation hyperparameter tunning
        cv_inner = KFold(n_splits=3, shuffle=True)        
        def treina_xgb(params):
            learning_rate = params[0]
            n_estimators = params[1]
            max_depth = params[2]
            min_child_weight = params[3]
            gamma = params[4]    
            subsample = params[5]
            colsample_bytree = params[6]
            model = xgb.XGBRegressor(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,
                                      min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,colsample_bytree=colsample_bytree)
            return -np.mean(cross_val_score(model,X_train,y_train,cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
        
        resultado_xgb = dummy_minimize(treina_xgb,space_xgb,n_calls=n_calls_hyp,verbose=1)
        param_xgb = resultado_xgb.x
        param_results.append(param_xgb)
        
    #Treino e teste do modelo
        xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                      min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                      subsample=param_xgb[5],colsample_bytree=param_xgb[6])

        xgbreg = xgb_reg.fit(X_train,y_train)
        xgb_pred = xgbreg.predict(X_test)
        xgb_pred_2 = xgbreg.predict(X_train)
    #Modelo de Random Forest para benchmark
        def random_forest_on(squared_rf,absolut_rf):
            def treina_random_forest(params):
                max_depth = params[0]
                min_samples_split = params[1]
                min_samples_leaf = params[2]
                model = RandomForestRegressor(n_estimators=1500,max_depth=max_depth,
                                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                return -np.mean(cross_val_score(model,X_train,y_train,cv=cv_inner,scoring="neg_mean_squared_error"))#mean_squared_error(y_test, p)
        
            resultado_random_forest = dummy_minimize(treina_random_forest,space_xgb,n_calls=n_calls_hyp,verbose=1)
            param_random = resultado_random_forest.x
            random_reg= RandomForestRegressor(n_estimators=1500,max_depth=param_random[0],
                                          min_samples_split=param_random[1], min_samples_leaf=param_random[2])

            randreg = random_reg.fit(X_train,y_train)
            rand_pred = randreg.predict(X_test)
            squared_rf.append(mean_squared_error(y_test, rand_pred)**0.5)
            absolut_rf.append(mean_absolute_error(y_test,rand_pred))
            return(squared_rf,absolut_rf)
        if train_rf==True: 
            squared_rf,absolut_rf = random_forest_on(squared_rf,absolut_rf)

        #Resultado da avaliação do modelo
        rmse.append(mean_squared_error(y_test, xgb_pred)**0.5)
        mae.append(mean_absolute_error(y_test,xgb_pred))
        rmse_treino.append(mean_squared_error(y_train, xgb_pred_2)**0.5)
        mae_treino.append(mean_absolute_error(y_train,xgb_pred_2))
        mape.append(np.mean(np.abs(y_test-xgb_pred)/y_test))

        if plot== True:
            ax = sns.histplot((y_test-xgb_pred)/y_test)
            ax.set(xlabel='Percentage error',ylabel='Frequência')
            plt.savefig('hist erro percentual.png')
            plot = False
    print('rolou')

    if train_rf == True:
        results_dic = {'MAE RF':absolut_rf,'RMSE RF':squared_rf,
                          'RMSE':rmse,'MAE':mae,
                           'MAPE':mape ,'RMSE treino':rmse_treino, 'MAE treino':mae_treino,
                          'XGB_Params':param_results
                          }
    else:
        results_dic = {'RMSE XGB':rmse,'MAE':mae,      
                       'MAPE':mape ,'RMSE treino':rmse_treino, 'MAE treino':mae_treino,
                          'XGB_Params':param_results
                          }
    return pd.DataFrame.from_dict(results_dic)

def final_model(df,df_pred,param_xgb,com_localizacao=True):
    #Modelo final para gerar previsões.
    from eli5 import show_weights, show_prediction
    from eli5.sklearn import PermutationImportance    
    import xgboost as xgb

    X = df[['sqm', 'bedrooms','bairro']]
    y = df['price_month']
    X_test = df_pred[['sqm','bedrooms','bairro']]
    
    if com_localizacao ==True:
        X, X_test = bairro(X,X_test,y)
    else:
        X.drop('bairro',axis=1,inplace=True)
    xgb_reg = xgb.XGBRegressor(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                  min_child_weight=param_xgb[3], gamma=param_xgb[4],subsample=param_xgb[5],
                                  colsample_bytree=param_xgb[6])
    xgbreg = xgb_reg.fit(X,y)
    pred = xgbreg.predict(X_test)
    
    
    # Explicação do modelo utilizando o eli5
    perm = PermutationImportance(xgbreg, scoring = 'neg_mean_absolute_error').fit(X, y)
    weights = show_weights(perm, feature_names = list(X_test.columns))
    show_1 = show_prediction(xgbreg,X_test.iloc[0],show_feature_values=True)
    show_2 = show_prediction(xgbreg,X_test.iloc[3],show_feature_values=True)
    X_test['pred'] = pred



    return X_test, weights, show_1, show_2

