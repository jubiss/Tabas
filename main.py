import pandas as pd
import sys
sys.path.append('/home/jubi/Documents/Processo seletivo /Tabas')
import functions as func

##################### IMPORTACAO E TRATAMENTO DE DADOS ####################

df = func.merge_data()
df = func.pre_processing(df)

#################### PARAMETROS PARA SEREM OTIMIZADOS #################
space_xgb = [(1e-3, 1e-1, 'log-uniform'), # learning rate
          (100, 2000), # n_estimators
          (1, 100), # max_depth 
          (1, 6.), # min_child_weight 
          (0, 15), # gamma 
          (0.5, 1.), # subsample 
          (0.5, 1.)] # colsample_bytree 

space_random = [(100,2000), #n_estimators
         (1,100), #max_depth
         (2,120), #min_sample_splits
         (1,10) #min_samples_leaf
         ]

#################### VALIDACAO MODELO #################

###Validacao modelo Long Stay
df_long = df[df['rent_type']=='long stay']
result_long = func.model_validation(df_long, space_xgb, space_random,train_rf=True,plot=True)
long_param = result_long[result_long['MAPE']==result_long['MAPE'].min()]['XGB_Params'].values[0]

###Validacao modelo Short Stay
df_short = df[df['rent_type']=='short stay']
result_short = func.model_validation(df_short, space_xgb, space_random,train_rf=True,plot=False)
short_param = result_short[result_short['MAPE']==result_short['MAPE'].min()]['XGB_Params'].values[0]

####################### PREDICAO TO PRICE #######################

to_price = pd.read_csv('to price.csv')

###Predicao Long Stay
model_long ,long_weights, long_show_1, long_show_2= func.final_model(df_long,to_price,long_param)
to_price['Long Stay'] = model_long['pred']

###Predicao Short Stay
model_short, short_weights, short_show_1, short_show_2 = func.final_model(df_short,to_price,short_param)
to_price['Short Stay']=model_short['pred']


to_price['Receita'] = to_price['Short Stay']-to_price['Long Stay']
to_price.to_csv('to_price_results.csv',index=False)

###################### PREDICAO DADOS RETIRADOS DO SITE DA TABAS #############

to_price_2 = pd.read_csv('apartamentos tabas.csv')

###Predicao Long Stay
model_long_2, long_weights_2, long_show_1_2, long_show_2_2 = func.final_model(df_long,to_price_2,long_param)
to_price_2['Long Stay']=model_long_2['pred']

###Predicao Short Stay
model_short_2, short_weights_2, short_show_1_2, short_show_2_2 = func.final_model(df_short,to_price_2,short_param)
to_price_2['Short Stay']=model_short_2['pred']


to_price_2['Receita'] = to_price_2['Short Stay']-to_price_2['Long Stay']
to_price_2.to_csv('apartamento_tabas_results.csv',index=False)