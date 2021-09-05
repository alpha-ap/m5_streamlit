import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

@st.cache
def load_data():
    test_df = pd.read_csv('preProcessedTest.csv')
    return test_df

test_df = load_data()

def getPredictions(input_id):
    global test_df
    with open('le_id.pkl', 'rb') as file:  
        le_id = pickle.load(file)
    
    ip_id = le_id.transform(np.array(input_id+'_evaluation').reshape(-1,1))[0]
    test_df = test_df.loc[test_df['id'] == ip_id]
    
    with open('Pickle_CAT.pkl', 'rb') as file:  
        cat = pickle.load(file)
    y_pred_cat = cat.predict(test_df)
    
    with open('Pickle_DT.pkl', 'rb') as file:  
        dt = pickle.load(file)
    y_pred_dt = dt.predict(test_df)    

    with open('Pickle_LGBM.pkl', 'rb') as file:  
        lgbm = pickle.load(file)
    y_pred_lgbm = lgbm.predict(test_df)
    
    with open('Pickle_RF.pkl', 'rb') as file:  
        rf = pickle.load(file)
    y_pred_rf = rf.predict(test_df)    

    with open('Pickle_XGB.pkl', 'rb') as file:  
        xgb = pickle.load(file)
    y_pred_xgb = xgb.predict(test_df)
    
    x_test_meta = np.vstack((y_pred_dt,y_pred_rf,y_pred_xgb,y_pred_lgbm,y_pred_cat)).T
    del y_pred_dt,y_pred_rf,y_pred_xgb,y_pred_lgbm,y_pred_cat
    with open('Pickle_LR.pkl', 'rb') as file:  
        lr = pickle.load(file)
    y_pred_lr = lr.predict(x_test_meta)
    
    lst = list(range(1,29))
    lst = ['Day '+str(x) for x in lst]
    d = {'id':le_id.inverse_transform(test_df['id'].values),'d':lst,'units':y_pred_lr}
    y_pred = pd.DataFrame(data=d)
    y_pred.drop(['id'],axis=1,inplace=True)
    del d
    return y_pred

def checkValidity(input_id):
    val = False
    with open('le_id.pkl', 'rb') as file:  
        le_id = pickle.load(file)
    try:
        le_id.transform(np.array(input_id+'_evaluation').reshape(-1,1))
        val = True
    except ValueError as ve:
        val = False
    return val


st.title("Walmart sales predicition")
st.subheader("Predicts next 28 days sales for input product and store id")
input_id = st.text_input(label="Enter ID in format: (ITEM_ID)_(STORE_ID)")


if st.button("Predict"):
    if checkValidity(input_id):
        pred_status = st.text('Getting predictions for '+input_id)
        st.dataframe(getPredictions(input_id),400,800)
        pred_status.text('Done!')
    else:
        st.write("Invalid ID. Please enter a valid ID")
	