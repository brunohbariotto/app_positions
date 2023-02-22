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
        
    def osc(self, prices, fast=32, slow=96):
        f,g = 1-1/fast, 1-1/slow
        return (prices.ewm(span=2*fast-1).mean() - prices.ewm(span=2*slow-1).mean())/np.sqrt(1.0 / (1 - f*f) - 2.0 / (1 - f*g) + 1.0 / (1 - g*g))
    
    def PL_osc(self,cotacoes, prices, volatility, fast=32, slow=96, fee = 30E-6):
        currency_position = (np.tanh(self.osc(prices, fast=fast, slow=slow)) / volatility).clip(-50, 50).fillna(0.0)
        euclid_norm = np.sqrt((currency_position*currency_position).sum(axis=1))
        
        currency_position_scaled = (currency_position.apply(lambda x : x/euclid_norm, axis=0)/volatility).clip(-50,50).fillna(0.0)
        profit_scaled = (cotacoes.pct_change() * currency_position_scaled.shift(periods=1).fillna(0.0))
        fee_scaled = np.abs(currency_position_scaled.diff()*fee).fillna(0.0)
        
        return currency_position_scaled, ((profit_scaled.subtract(fee_scaled))/volatility), (1.0+profit_scaled.subtract(fee_scaled)/volatility/(50*profit_scaled.shape[1])).fillna(1.0)
        
        
    def oscilador(self, df_prices):
        st.subheader('Oscilador')
        
        st.markdown('---')
        
        volatility = np.log(df_prices.ffill()).diff().ewm(com=32).std()
        prices = (np.log(df_prices.ffill()).diff() / volatility).clip(-5,5).cumsum()
        
        pos_slow, pnl_slow, pnl_comp_slow = self.PL_osc(df_prices, prices, volatility, fast=32, slow=96)
        pos_medium, pnl_medium, pnl_comp_medium = self.PL_osc(df_prices, prices, volatility, fast=16, slow=48)
        pos_fast, pnl_fast, pnl_comp_fast = self.PL_osc(df_prices, prices, volatility, fast=8, slow=24)
        
        pos_slow_q = pd.DataFrame(columns=pos_slow.columns)
        pos_medium_q = pd.DataFrame(columns=pos_medium.columns)
        pos_fast_q = pd.DataFrame(columns=pos_fast.columns)
        
        pos_slow_p = pd.DataFrame(index = pos_slow.index ,columns=pos_slow.columns)
        pos_medium_p = pd.DataFrame(index = pos_medium.index ,columns=pos_medium.columns)
        pos_fast_p = pd.DataFrame(index = pos_fast.index ,columns=pos_fast.columns)
        
        pos_slow_q.loc['q3',:] = pos_slow.quantile(.75)
        pos_slow_q.loc['q2',:] = pos_slow.median()
        pos_slow_q.loc['q1',:] = pos_slow.quantile(.25)
        
        pos_medium_q.loc['q3',:] = pos_medium.quantile(.75)
        pos_medium_q.loc['q2',:] = pos_medium.median()
        pos_medium_q.loc['q1',:] = pos_medium.quantile(.25)
        
        pos_fast_q.loc['q3',:] = pos_fast.quantile(.75)
        pos_fast_q.loc['q2',:] = pos_fast.median()
        pos_fast_q.loc['q1',:] = pos_fast.quantile(.25)
        
        pos_slow_p.iloc[:,:] = np.where(pos_slow.iloc[:,:] >= pos_slow_q.loc['q3',:], 0,
                          np.where( (pos_slow.iloc[:,:] >= pos_slow_q.loc['q2',:]) & (pos_slow.iloc[:,:] < pos_slow_q.loc['q3',:]), 15,
                          np.where( (pos_slow.iloc[:,:] >= pos_slow_q.loc['q1',:]) & (pos_slow.iloc[:,:] < pos_slow_q.loc['q2',:]), 30, 50 )))
        
        st.write('Posição Lenta')
        st.write(pos_slow_p)
        
        pos_medium_p.iloc[:,:] = np.where(pos_medium.iloc[:,:] >= pos_medium_q.loc['q3',:], 0,
                          np.where( (pos_medium.iloc[:,:] >= pos_medium_q.loc['q2',:]) & (pos_medium.iloc[:,:] < pos_medium_q.loc['q3',:]), 10,
                          np.where( (pos_medium.iloc[:,:] >= pos_medium_q.loc['q1',:]) & (pos_medium.iloc[:,:] < pos_medium_q.loc['q2',:]), 25, 45 )))

        st.write('Posição Média')
        st.write(pos_medium_p)
        
        pos_fast_p.iloc[:,:] = np.where(pos_fast.iloc[:,:] >= pos_fast_q.loc['q3',:], 0,
                          np.where( (pos_fast.iloc[:,:] >= pos_fast_q.loc['q2',:]) & (pos_fast.iloc[:,:] < pos_fast_q.loc['q3',:]), 5,
                          np.where( (pos_fast.iloc[:,:] >= pos_fast_q.loc['q1',:]) & (pos_fast.iloc[:,:] < pos_fast_q.loc['q2',:]), 20, 40 )))
        
        st.write('Posição Rápida')
        st.write(pos_fast_p)
        
        ultimo_preco = df_prices.iloc[-1,:]
        st.write(ultimo_preco)
        
        fig_preco = go.Figure()
        fig_vol = go.Figure()
        fig_pos = go.Figure()
        fig_pos_box_slow = go.Figure()
        fig_pos_box_medium = go.Figure()
        fig_pos_box_fast = go.Figure()
        fig_PL = go.Figure()
        fig_PLAcc = go.Figure()
        for ac in pos_slow.columns:
            
            fig_preco.add_trace(go.Scatter(
                 x=df_prices.index,
                 y=df_prices[ac],
                 name=ac))
            fig_preco.update_layout(
                title={'text':'Preço'},
                xaxis_title='Data',
                yaxis_title='Preço (R$)',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            fig_vol.add_trace(go.Scatter(
                 x=volatility.index,
                 y=volatility[ac],
                 name=ac))
            fig_vol.update_layout(
                title={'text':'Volatilidade'},
                xaxis_title='Data',
                yaxis_title='Vol',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            fig_pos_box_medium.add_trace(go.Box(
                y=pos_slow[ac],
                name=ac)
                )
                 
            fig_pos_box_slow.update_layout(
                title={'text':'Posição Escalada BoxPlot - SLOW'},
                xaxis_title='Ticker',
                yaxis_title='Posição',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            
            
            fig_pos_box_medium.add_trace(go.Box(
                y=pos_medium[ac],
                name=ac)
                )

            fig_pos_box_medium.update_layout(
                title={'text':'Posição Escalada BoxPlot - MEDIUM'},
                xaxis_title='Ticker',
                yaxis_title='Posição',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            fig_pos_box_fast.add_trace(go.Box(
                y=pos_fast[ac],
                name=ac)
                )

            fig_pos_box_fast.update_layout(
                title={'text':'Posição Escalada BoxPlot - FAST'},
                xaxis_title='Ticker',
                yaxis_title='Posição',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            
        st.plotly_chart(fig_preco)
        st.plotly_chart(fig_vol)
        
        fig_pos_box_slow.add_trace(go.Scatter(
            x=pos_slow.columns,
            y=pos_slow.iloc[-1,:])
            )
        
        
        st.plotly_chart(fig_pos_box_slow)
        
        
        fig_pos_box_medium.add_trace(go.Scatter(
            x=pos_medium.columns,
            y=pos_medium.iloc[-1,:])
            )
        
        
        st.plotly_chart(fig_pos_box_medium)
        
        
        fig_pos_box_fast.add_trace(go.Scatter(
            x=pos_fast.columns,
            y=pos_fast.iloc[-1,:])
            )
        
        
        st.plotly_chart(fig_pos_box_fast)
        
        final_pos_osc = pd.DataFrame(columns=pos_fast_p.columns)
        final_pos_osc = (pos_fast_p + pos_medium_p + pos_slow_p).iloc[-1,:]

        st.write('Posição Final [%] - Histórico')
        st.write(pos_fast_p + pos_medium_p + pos_slow_p)
        
        st.write('Posição Final [%] - Mais Recente')
        final_pos_osc.loc['HASH11.SA',:] = final_pos_osc.loc['BTC-USD',:]
        st.write(final_pos_osc)
        
    def markowitz(self):
        st.subheader('Markowitz')
        
        st.markdown('---')