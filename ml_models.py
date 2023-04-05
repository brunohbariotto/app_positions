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
            
            self.description_count()
            
            
        elif self.model == "Regressão Logística":
            st.markdown('---')
            st.header('**Modelos de Regressão Logística**')
            
        elif self.model == "PCA":
            st.markdown('---')
            st.header('**Principal Component Analysis**')
            
        elif self.model == "Clusterização":
            st.markdown('---')
            st.header('**Modelos de Clusterização**')
            
            
    def description_count(self):
        st.subheader('DataFrame')
        st.write(self.df)
        
        st.subheader('Info')
        st.write(self.df.info())
        
        st.subheader('Describe')
        st.write(self.df.describe())
        
        st.subheader('Histograma de Contagem')
        contagem = self.df[self.y_var].value_counts(dropna=False)
        percent = self.df[self.y_var].value_counts(dropna=False, normalize=True)
        pd.concat([contagem, percent], axis=1, keys=['count', '%'], sort=True)
        
        plt.figure(figsize=(15,20))
        sns.histplot(data=self.df, x=self.y_var, bins=20, color='darkorchid')
        plt.show()
        
        st.subheader('Média e Variância de Y')
        st.write(pd.DataFrame({'Mean':[self.df[self.y_var].mean()],
                               'Variance':[self.df[self.y_var].var()]}))
        
        
        
        
        
            
        
