# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:00 2023

@author: Dell inspiron
"""

import streamlit as st

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
            
        elif self.model == "Regressão Logística":
            st.markdown('---')
            st.header('**Modelos de Regressão Logística**')
            
        elif self.model == "PCA":
            st.markdown('---')
            st.header('**Principal Component Analysis**')
            
        elif self.model == "Clusterização":
            st.markdown('---')
            st.header('**Modelos de Clusterização**')
            
        
