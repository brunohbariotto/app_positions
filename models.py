# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:09:49 2023

@author: Dell inspiron
"""

import streamlit as st
import yfinance as yf
import pandas as pd

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
        return df.pct_change()