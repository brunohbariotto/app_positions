# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:00 2023

@author: Dell inspiron
"""

import streamlit as st
import matplotlib.pyplot as plt

class Ml_models:
    def __init__(self, model):
        self.model = model
        
    def choose_model(self):        
        
        if self.model == "Regressão Linear":
            st.markdown('---')
            st.header('**Modelos de Regressão**')
            
        elif self.model == "Contagem":
            st.markdown('---')
            st.header('**Modelos de Contagem**')

            st.markdown("**Y**: **Quantitativa** com **Valores Inteiros** e **não negativos**")
            st.markdown("**Modelos de Regressão**: **Poisson** e **Binomial Negativo**")
            
            st.subheader("**Poisson**")
            st.latex(r'''\ln(\hat{Y}) = \alpha + \beta_{1}.X_{1}$]''')
            st.latex(r'''ln(\hat{Y} = \alpha + \beta_{1}.X_{1} + ... + \beta_{k}.X_{k}''')
            st.markdown("**Probabilidade** de ocorrência de uma contagem m em dada exposição")
            st.latex('''p(Y_{i} = m) = \frac{e^{-\lambda}.\lambda^{m}}{m!}''')
            st.markdown("em que $\lambda$ é o número esperado de ocorrências ou taxa média estimada de incidências")
            st.markdown("Para **Poisson**, Var $\approx $ Média = $\mu = \lambda_{poisson} $")
            
            
            
        elif self.model == "Regressão Logística":
            st.markdown('---')
            st.header('**Modelos de Regressão Logística**')
            
        elif self.model == "PCA":
            st.markdown('---')
            st.header('**Principal Component Analysis**')
            
        elif self.model == "Clusterização":
            st.markdown('---')
            st.header('**Modelos de Clusterização**')
            
        
