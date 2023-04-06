# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:00 2023

@author: Dell inspiron
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf #poisson and binneg model
import statsmodels.api as sm #poisson and binneg family
from statsmodels.iolib.summary2 import summary_col
#from statstests.tests import overdisp


class Ml_models:
    def __init__(self, model, df, y_var, x_var):
        self.model = model
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        
    def choose_model(self):        
        
        if self.model == "Regressão Linear":
            st.markdown('---')
            st.header('**Modelos de Regressão**')
            
        elif self.model == "Contagem":
            st.markdown('---')
            st.header('**Modelos de Contagem**')
            
            st.markdown("**Modelos de Regressão**: **Poisson** e **Binomial Negativo**")

            st.markdown("**Y (Variável Dependente)**: Quantitativa com Valores Inteiros e não negativos")
            
            df_cont = self.df.copy()
            
            df_cont[self.x_var] = df_cont[self.x_var].astype('float')
            df_cont[self.y_var] = df_cont[self.y_var].astype('int64')
            formula = f'{self.x_var[0]} ~ ' + ' + '.join(self.x_var)
            
            #Description
            self.description_count(df_cont, self.y_var[0], self.x_var)
            
            #Poisson Model
            
            m_poi = self.poisson(df_cont, self.y_var[0], self.x_var, formula)
            
            
            st.subheader('Summary')
            st.write(m_poi.summary())
            
            
            st.subheader('Predict')
            
            st.write('Insira os Parâmetros')
            df_pred = pd.pivot_table(df_cont, index='Input', values=df_cont.loc[:,self.x_var].iloc[-1,:].values, columns=self.x_var)
            #df_pred = df_cont.copy().loc[:,self.x_var].iloc[-1,:]
            #df_pred.columns = ['Inputs']
            st.write(df_pred)
            df_pred_input = st.experimental_data_editor(df_pred, num_rows="dynamic")
            
            df_pred_output = m_poi.predict(df_pred_input)
            df_pred_output.columns = ['Output_Poisson']
            
            st.write(df_pred_output)
            
            
            
            
        elif self.model == "Regressão Logística":
            st.markdown('---')
            st.header('**Modelos de Regressão Logística**')
            
        elif self.model == "PCA":
            st.markdown('---')
            st.header('**Principal Component Analysis**')
            
        elif self.model == "Clusterização":
            st.markdown('---')
            st.header('**Modelos de Clusterização**')
            
            
    def description_count(self, df, y_var, x_var):
        st.markdown('---')
        st.header('**Description**')
        st.subheader('DataFrame')
        st.write(df)
        
        st.subheader('Info')
        st.write(df.info())
        
        st.subheader('Describe')
        st.write(df.describe())
        
        st.subheader('Histograma de Contagem')
        contagem = df[y_var].value_counts(dropna=False)
        percent = df[y_var].value_counts(dropna=False, normalize=True)
        st.write(pd.concat([contagem, percent], axis=1, keys=['count', '%'], sort=True))
        
        n_bins = st.number_input('Bins', min_value=1, value=20, step=1, key=13175)
        
        fig_hist_count = plt.figure(figsize=(20,15))
        sns.histplot(data=df, x=y_var, bins=n_bins, color='darkorchid')
        st.pyplot(fig_hist_count)
        
        st.subheader('Média e Variância de Y')
        st.write(pd.DataFrame({'Mean':[df[y_var].mean()],
                               'Variance':[df[y_var].var()]}))
        
        
        
    def poisson(self, df, y_var, x_var, formula):
        st.markdown('---')
        st.header("**Poisson**")
        st.latex(r'''ln(\hat{Y}) = \alpha + \beta_{1}.X_{1} + ... + \beta_{k}.X_{k}''')
        st.markdown("**Probabilidade** de ocorrência de uma contagem m em dada exposição")
        st.latex(r'''p(Y_{i} = m) = \left(\frac{e^{-\lambda}.\lambda^{m}}{m!}\right)''')
        st.markdown("em que $\lambda$ é o número esperado de ocorrências ou taxa média estimada de incidências")
        st.latex( r'''Var \approx Média = \mu = \lambda_{poisson}''')
        
        m_poisson = smf.glm(formula=formula, 
                            data=df,
                            family = sm.families.Poisson()).fit()
        
        return m_poisson

        
            
        
