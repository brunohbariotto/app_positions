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
import pypfopt
# from PyPortfolioOpt import expected_returns
# from PyPortfolioOpt import risk_models
# from PyPortfolioOpt import EfficientFrontier
# from PyPortfolioOpt import objective_functions
# from PyPortfolioOpt.discrete_allocation import DiscreteAllocation, get_latest_prices
# from PyPortfolioOpt import HRPOpt
# from PyPortfolioOpt import plotting
import matplotlib.pyplot as plt

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
        
        #st.write('Posição Lenta')
        #st.write(pos_slow_p)
        
        pos_medium_p.iloc[:,:] = np.where(pos_medium.iloc[:,:] >= pos_medium_q.loc['q3',:], 0,
                          np.where( (pos_medium.iloc[:,:] >= pos_medium_q.loc['q2',:]) & (pos_medium.iloc[:,:] < pos_medium_q.loc['q3',:]), 10,
                          np.where( (pos_medium.iloc[:,:] >= pos_medium_q.loc['q1',:]) & (pos_medium.iloc[:,:] < pos_medium_q.loc['q2',:]), 25, 45 )))

        #st.write('Posição Média')
        #st.write(pos_medium_p)
        
        pos_fast_p.iloc[:,:] = np.where(pos_fast.iloc[:,:] >= pos_fast_q.loc['q3',:], 0,
                          np.where( (pos_fast.iloc[:,:] >= pos_fast_q.loc['q2',:]) & (pos_fast.iloc[:,:] < pos_fast_q.loc['q3',:]), 5,
                          np.where( (pos_fast.iloc[:,:] >= pos_fast_q.loc['q1',:]) & (pos_fast.iloc[:,:] < pos_fast_q.loc['q2',:]), 20, 40 )))
        
        #st.write('Posição Rápida')
        #st.write(pos_fast_p)
        
        ultimo_preco = df_prices.iloc[-1,:]
        #st.write(ultimo_preco)
        
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
                 y=volatility[ac]*100,
                 name=ac))
            fig_vol.update_layout(
                title={'text':'Volatilidade'},
                xaxis_title='Data',
                yaxis_title='Vol [%]',
                font=dict(
                    family="Courier New, monospace",
                    size=14))
            
            fig_pos_box_slow.add_trace(go.Box(
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
        st.write(final_pos_osc)
        st.write(df_prices)

        if 'BTC-USD' in df_prices.columns.get_level_values(0):
            st.write('Entrou')
            final_pos_osc.loc['HASH11.SA'] = final_pos_osc.loc['BTC-USD']
        else:
            st.write('Insira BTC-USD para ponderar HASH11')
        #final_pos_osc.loc['HASH11.SA'] = final_pos_osc.loc['BTC-USD']
        
        return final_pos_osc.to_frame()
    
    def markowitz(self, precos, volatilidade ,anos_cotacoes, otimizador, cov_type, is_longOnly, regul_zeros, exp_return_type,
                  vol_effic=0.2, ret_effic=0.2
                         , span=100, selic_aa=0.02
                         , span_cov=100, cash=100000):
        
        url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json'
        df = pd.read_json(url)
        
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df.set_index('data', inplace=True)
        df = df['1996-01-01':]
        
        df['Selic'] = ((1+df['valor'].iloc[:]/100)**(252) -1)*100
        selic_diaria = (1+selic_aa)**(1/252) -1
        
        #retorno anualizado
        precos.dropna(inplace=True)
        cf_anual = (precos.iloc[-1]-precos.iloc[0])/precos.iloc[0]
        cf_anual = ((1+cf_anual)**(1/anos_cotacoes))-1
        
        num_ativos = len(precos.columns)
        pesos = np.array(num_ativos*[1/num_ativos])
        
        ret_anual = cf_anual.dot(pesos)
        
        
        #vol da carteira
        cov = precos.pct_change().cov()
        vol_diaria = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        vol_anual = vol_diaria*np.sqrt(252)
        
        
        ### --- Expectativa de Retornos
        
        if exp_return_type == 'mean_historical_return':
            #retorno medio historico
            mu = pypfopt.expected_returns.mean_historical_return(precos)
        if exp_return_type == 'ema_historical_return':
        #retorno media movel exponencial
            mu = pypfopt.expected_returns.ema_historical_return(precos, span=span)
        if exp_return_type == 'capm':
        #retorno CAPM - retorno benchmark (IBOV) > CAPM = Rf + beta*(Rm-Rf)
            
            ibov = pd.DataFrame(yf.download('^BVSP', period='f{anos_cotacoes}y')['Adj Close'])
            mu = pypfopt.expected_returns.capm_return(precos, market_prices=ibov, risk_free_rate=df['Selic'].mean())
        
        #'CovarianceShrinkage', 'sample_cov', 'semicovariance', 'exp_cov'
        ### --- Matrizes de covariância
        # Matriz de covariancia
        if cov_type == 'sample_cov':
            cov = pypfopt.risk_models.sample_cov(precos)
        if cov_type == 'semicovariance':
        # Matriz de Semicovariancia - benchmark = 0 pega somente retornos abaixo de 0 (perdas)
        # Somente modelos que aceitem matriz de semicov
            cov = pypfopt.risk_models.semicovariance(precos, benchmark=0)
        if cov_type == 'exp_cov':
        #Exponentially-Weighted Covariance - peso maior para informações mais recentes
            cov = pypfopt.risk_models.exp_cov(precos, span=200)
        if cov_type == 'CovarianceShrinkage':
        # Estimadores de Ledoit Wolf - Redução de valores extremos (normalização da matriz de cov)
        # constant_variance : diagonal da matriz como media das variancias dos retornos
        # single_factor: baseado no sharp, utiliza o beta como parâmetro do encolhedor
        # constant_correlation: relacionado a matriz de correlação e desvio da amostra
            cov = pypfopt.risk_models.CovarianceShrinkage(precos).ledoit_wolf()
        
        
        #'MaxSharp','MinVol','EfficientRisk','EfficientReturn','RiskParity'
        ### --- Modelos de Otimização
        # Modelo de Portólio de Mínima Variancia > Reduzir a Volatilidade do Portfólio, pesos = 1, long_only
        # Se long short (weight-bounds(none,none))
        if is_longOnly == 'Sim':
            mv = pypfopt.EfficientFrontier(mu, cov)
        if is_longOnly == 'Não':
            mv = pypfopt.EfficientFrontier(mu, cov, weight_bounds=(-1,1))
            
        if regul_zeros == 'Sim':
            mv.add_objective(pypfopt.objective_functions.L2_reg, gamma=0.1)
        
        if otimizador == 'MinVol':
            w = mv.min_volatility()
            
        if otimizador == 'MaxSharp':
            w = mv.max_sharpe()
        
        if otimizador == 'EfficientRisk':
            w = mv.efficient_risk(target_volatility=vol_effic)
            
        if otimizador == 'EfficientReturn':
            w = mv.efficient_return(target_return=ret_effic)
            
        if otimizador == 'RiskParity':
            hrp_portfolio = pypfopt.HRPOpt(pypfopt.expected_returns.returns_from_prices(precos))
            cleaned_weights = hrp_portfolio.optimize()
            model_ret = hrp_portfolio.portfolio_performance(verbose=True, risk_free_rate= (1+df['Selic'].iloc[-1])**(1/252) -1 )
            #st.write(hrp_portfolio)
            fig = plt.figure()
            pypfopt.plotting.plot_dendrogram(hrp_portfolio)
            st.pyplot(fig)
            st.set_option('deprecation.showPyplotGlobalUse', True)
            
        if otimizador != 'RiskParity':
            cleaned_weights = mv.clean_weights()
            model_ret = mv.portfolio_performance(verbose=True, risk_free_rate=df['Selic'].iloc[-1]/100)

        pesos = np.array(list(cleaned_weights.values()))
        
        vol_otimizada = np.sqrt(np.dot(pesos.T, np.dot(cov,pesos)))
        ret_otimizado = cf_anual.dot(pesos)
        
        
        ultimos_precos = pypfopt.get_latest_prices(precos)

        
        da = pypfopt.DiscreteAllocation(cleaned_weights, ultimos_precos, total_portfolio_value=cash)
        
        if otimizador != 'RiskParity':
            allocation, leftover = da.lp_portfolio()
        elif otimizador == 'RiskParity':
            allocation, leftover = da.greedy_portfolio()
            
        alocacao = pd.DataFrame(allocation.values(),index=allocation.keys())
        
        final_dataframe = pd.DataFrame(cleaned_weights.values(),index=cleaned_weights.keys())
        final_dataframe = pd.concat([final_dataframe, alocacao.iloc[:,0]], axis=1)
        final_dataframe = pd.concat([final_dataframe, ultimos_precos], axis=1)
        final_dataframe.columns = ['Pesos %', 'Pos. Qtdd', 'Últ. Preço']
        final_dataframe['Pesos %'] = final_dataframe['Pesos %']*100
        final_dataframe['Pos. R$'] = final_dataframe['Pos. Qtdd']*final_dataframe['Últ. Preço']
        
        final_dataframe['Vol. Média Anual'] = volatilidade.mean()*np.sqrt(252)
        
        return ret_anual, vol_anual, final_dataframe.fillna(0).round({'Pesos %':2, 'Pos. Qtdd':0, 'Últ. Preço':2}), ret_otimizado, vol_otimizada, model_ret
    
        
    def markowitz_inputs(self, df_prices,anos_cotacoes):
        
        st.subheader('Markowitz')
        
        st.markdown('---')
        
        volatility = np.log(df_prices.ffill()).diff().ewm(com=32).std()
        prices = (np.log(df_prices.ffill()).diff() / volatility).clip(-5,5).cumsum()
        
        vol_effic, ret_effic, span, selic_aa, span_cov = 0.2,0.2,100,0.02,100       

        otimizador = st.radio('Escolha a Otimização', ['MaxSharp','MinVol','EfficientRisk','EfficientReturn','RiskParity'])
        
        cash = st.number_input('Coloque o valor alocado em R$', min_value=100, step=100, key=1974)
        
        if otimizador == 'EfficientRisk':
            vol_effic = st.number_input('Qual a Volatilidade desejada', min_value=0.05, key=111)
        if otimizador == 'EfficientReturn':
            ret_effic = st.number_input('Qual o Retorno desejado', min_value=0.05, key=222)
        
        exp_return_type = st.radio('Escolha o Tipo de Expected Return', ['ema_historical_return', 'mean_historical_return', 'capm'])
        if exp_return_type == 'ema_historical_return':
            span = st.number_input('Escolha o Span', min_value=100, key=333)
            
        cov_type = st.radio('Escolha o Tipo de Covariância', ['CovarianceShrinkage', 'sample_cov', 'semicovariance', 'exp_cov'])
        if cov_type == 'exp_cov':
            span_cov = st.number_input('Escolha o Span', min_value=100, key=555)
            
        is_longOnly = st.radio('Estratégia Long Only?', ['Sim', 'Não'])
        
        regul_zeros = st.radio('Regularização remoção de nulos?', ['Sim', 'Não'])
        
        ret_anual, vol_anual, pesos, ret_otimizado, vol_otimizada, model_ret =  self.markowitz(df_prices, volatility ,anos_cotacoes, 
                             otimizador, cov_type, is_longOnly, regul_zeros, exp_return_type,
                                           vol_effic, ret_effic
                                                  , span, selic_aa
                                                  , span_cov,cash)
        
        st.write(pesos)
        st.write(pd.DataFrame(model_ret, index=['Expected annual return','Annual volatility','Sharpe Ratio'], columns=['Values']))