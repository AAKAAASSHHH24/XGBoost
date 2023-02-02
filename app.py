
import streamlit as st
from utils import columns, cat_cols, num_cols

import numpy as np
import pandas as pd
import pickle
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

#st.set_option('server.host', '0.0.0.0')
#st.set_option('server.port', 8080)

st.set_page_config(page_title="WILL THE INCOME BE MORE THAN 50K DOLLARS?",
                 layout="wide")

#model = joblib.load('XG_Boost_job.joblib')

model = xgb.XGBClassifier()
model.load_model("model_sklearn.json")


# ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week',	'native_country']



opt_workclass= ['Private', 'Self-emp-not-inc', 'Local-gov','State-gov','Self-emp-inc','Federal-gov', 'Without-pay', 'Never-worked']
opt_education = ['HS-grad','Some-college','Bachelors', 'Masters','Assoc-voc', '11th', 'Assoc-acdm',
                                                 '10th', '7th-8th','Prof-school','9th','12th','Doctorate','5th-6th','1st-4th',
                                                 'Preschool']
opt_education_num = [13,  9,  7, 14,  5, 10, 12, 11,  4, 16, 15,  3,  6,  2,  1,  8]
opt_marital_status =[' Never-married', ' Married-civ-spouse', ' Divorced',
       ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
       ' Widowed']
opt_occupation =[' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
       ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
       ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
       ' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces',
       ' Priv-house-serv']
opt_relationship =[' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
       ' Other-relative']
opt_race = [' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
       ' Other']
opt_sex =[' Male', ' Female']
opt_native_country = [' United-States', ' Cuba', ' Jamaica', ' India', ' ?', ' Mexico',
       ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada',
       ' Germany', ' Iran', ' Philippines', ' Italy', ' Poland',
       ' Columbia', ' Cambodia', ' Thailand', ' Ecuador', ' Laos',
       ' Taiwan', ' Haiti', ' Portugal', ' Dominican-Republic',
       ' El-Salvador', ' France', ' Guatemala', ' China', ' Japan',
       ' Yugoslavia', ' Peru', ' Outlying-US(Guam-USVI-etc)', ' Scotland',
       ' Trinadad&Tobago', ' Greece', ' Nicaragua', ' Vietnam', ' Hong',
       ' Ireland', ' Hungary', ' Holand-Netherlands']



features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
       'marital_status', 'occupation', 'relationship', 'race', 'sex',
       'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
       'wage_class']

# take input 
st.markdown("<h1 style='text-align: center;'>WILL THE INCOME BE MORE THAN 50K DOLLARS? 🚧</h1>", unsafe_allow_html=True)

def main():
    
    with st.form("salary_prediction"):
        st.subheader("Pleas enter the following inputs:")
        education = st.selectbox("Level of education?:", options=opt_education)
        workclass = st.selectbox("Level of education?:", options=opt_workclass)
        marital_status = st.selectbox("Level of education?:", options=opt_marital_status)
        occupation = st.selectbox("marital_status?:", options=opt_occupation)
        relationship = st.selectbox("RELATIONSHIP?:", options=opt_relationship)
        race = st.selectbox("RACE?:", options=opt_race)
        sex = st.selectbox("SEX?:", options=opt_sex)
        native_country = st.selectbox("Native_country?:", options=opt_native_country)
        fnlwgt = st.number_input("Enter fnlwgt:")
        capital_gain = st.number_input("Enter capital gain:",0,99999)
        age = st.number_input("Enter age:",0,100)
        education_num = st.selectbox("Number code of education?:", options=opt_education_num)
        capital_loss = st.number_input("Enter capital loss:",0,99999)
        hours_per_week = st.slider("Number of working hours per week:",1,100, value=0, format="%d")
        
           
        submit = st.form_submit_button("Predict")
        
    if submit:
        numeric_pipeline = Pipeline(steps=[("scale", StandardScaler())])
        categorical_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")),
                                                ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),])
        full_processor = ColumnTransformer(transformers=[("numeric", numeric_pipeline, num_cols),
                                                            ("categorical", categorical_pipeline, cat_cols)])
        
        input_array = np.array([age,workclass,fnlwgt,education,
                                education_num,marital_status,occupation,relationship,
                                race,sex,capital_gain,capital_loss,hours_per_week,native_country], ndmin=2)
        
        X_processed_test = full_processor.fit_transform(input_array) 
        # Make predictions on the test data
        y_pred = model.predict(X_processed_test)
        
        
        if y_pred[0] == 0:
            st.write(f"The expected salary is less than 50000 dollars annualy")
        elif y_pred[0] == 1:
            st.write(f"The expected salary is more than 50000 dollars annualy")
            
        st.write("Developed By: Akash kumar Rakshit")
        st.markdown("""Reach out to me on: [LinkedIN](https://www.linkedin.com/in/akash-rakshit-020761175/)""")
              
            

st.subheader("🧾Description:")
st.text("""DataSet Information:

Extraction was done by Barry Becker from the 1994 Census
database. A set of reasonably clean records was extracted using the
following conditions: ((AAGE>16) && (AGI>100) &&
(AFNLWGT>1)&& (HRSWK>0))""")

st.markdown("Please find GitHub repository link of project: [Click Here](https://github.com/AAKAAASSHHH24/XGBoost)") 


if __name__ == '__main__':
   main()


