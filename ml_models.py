# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:00 2023

@author: Dell inspiron
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
            
            st.subheader("**Modelos de Regressão**: **Poisson** e **Binomial Negativo**")

            st.markdown("**Y (Variável Dependente)**: Quantitativa com Valores Inteiros e não negativos")
            
            st.markdown('---')
            st.header("**Poisson**")
            st.latex(r'''ln(\hat{Y}) = \alpha + \beta_{1}.X_{1} + ... + \beta_{k}.X_{k}''')
            st.markdown("**Probabilidade** de ocorrência de uma contagem m em dada exposição")
            st.latex(r'''p(Y_{i} = m) = \left(\frac{e^{-\lambda}.\lambda^{m}}{m!}\right)''')
            st.markdown("em que $\lambda$ é o número esperado de ocorrências ou taxa média estimada de incidências")
            st.latex( r'''Var \approx Média = \mu = \lambda_{poisson}''')
            
            self.description_count(self.df, self.y_var[0], self.x_var)
            
            
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
        
        fig_hist_count = plt.figure(figsize=(20,15))
        sns.histplot(data=df, x=y_var, bins=15, color='darkorchid')
        st.pyplot(fig_hist_count)
        
        st.write(df[y_var])
        #st.write(df[y_var].values.var())
        
        st.subheader('Média e Variância de Y')
        st.write(pd.DataFrame({'Mean':[df[y_var].astype('int64').mean()],
                               'Variance':[df[y_var].astype('int64').var()]}))
        
        
        
        
        
            
        
