# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:09:49 2023

@author: Dell inspiron
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

class Models:
    def __init__(self):
        pass
    
    def download_prices(self, ativos,per_data, anos_cotacoes, datas_inicio, datas_fim):
        cotacoes = pd.DataFrame()
        with st.spinner('Baixando Cotações...'):
            if per_data == 'Períodos':
                cotacoes = pd.concat([yf.download(tick , period=f'{anos_cotacoes}y')['Adj Close'] for tick in ativos], axis=1)
                cotacoes.columns = ativos
            if per_data == 'Data':
                cotacoes = pd.concat([yf.download(tick, start=datas_inicio , end=datas_fim)['Adj Close'] for tick in ativos], axis=1)
                cotacoes.columns = ativos
            if per_data == 'Máx':
                cotacoes = pd.concat([yf.download(tick, period='max')['Adj Close'] for tick in ativos], axis=1)
                cotacoes.columns = ativos 
        return cotacoes.fillna(method='ffill')
    
    def returns(self, df):
        
        if 'HASH11.SA' in df.columns:
            df.drop(columns=['HASH11.SA'], inplace=True)
            
        return df.pct_change()
    
    def correlacao(self, df_prices):

        st.subheader('Correlação')
        
        st.markdown('---')
        st.subheader('Matriz de Correlação')
        
        df_corr = df_prices.pct_change().corr()
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Heatmap(
            x=df_corr.columns,
            y=df_corr.index,
            z=np.array(df_corr),
            text=df_corr.values,
            texttemplate='%{text:.2f}',
            colorscale='RdBu'))
        
        fig_m.layout.height = 800
        fig_m.layout.width = 800
        
        st.plotly_chart(fig_m)
        
    def oscilador(self):
        st.subheader('Oscilador')
        
        st.markdown('---')
        
    def markowitz(self):
        st.subheader('Markowitz')
        
        st.markdown('---')