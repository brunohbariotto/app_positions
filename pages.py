# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:34:08 2023

@author: Dell inspiron
"""

import streamlit as st
import plotly.graph_objects as go
from models import Models
from PIL import Image
import investpy as inv
import pandas as pd
from datetime import datetime, date
import numpy as np
from ta.trend import SMAIndicator
import yfinance as yf


class Pages:
    def __init__(self):
        pass
        
    # posicao
    # Tela que exibe a posição atual real bem como sua comparação com a posição de Markwitz e Oscilator
    #
    def posicao(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.title('Controle de Posição')
        st.markdown('---')
        #image = Image.open('alocation2.jpg')
        #st.image(image, caption='Alocação',width=400)
        #st.dataframe(df)
        
        m = Models()
        df_prices = m.download_prices(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
        #st.write(df_prices)
        
        #Pizza da posição atual
        st.subheader('Posição Atual [% e R$]')
        fig = go.Figure(data=[go.Pie(labels=df.Acao, values=(df_prices.iloc[-1,:].T * df.pos_atual.values), textinfo='label+percent+value')])
        fig.layout.height = 700
        fig.layout.width = 700
        
        fig.update_layout(
            title={'text':'Posição Atual [% e R$]'},
            font=dict(
                family="Courier New, monospace",
                size=12))
        
        st.plotly_chart(fig)
        
        #Comparação Posição atual x Markowitz x Oscilator
        st.markdown('---')
        st.subheader('Comparação entre as Posições: Markowitz, Oscilator e Atual')
        fig2 = go.Figure(data=[
            go.Bar(name='Pos. Markowitz', x=df.Acao, y= df.pos_markw),
            go.Bar(name='Pos. Oscilator', x=df.Acao, y= df.pos_osc*df.pos_markw/100),
            go.Bar(name='Pos. Atual', x=df.Acao, y= df.pos_atual)
        ])
        fig2.layout.height = 700
        fig2.layout.width = 800
        
        fig2.update_layout(
            title={'text':'Comparação entre as Posições: Markowitz, Oscilator e Atual'},
            xaxis_title='Ações',
            yaxis_title='Quantidade de Ações',
            font=dict(
                family="Courier New, monospace",
                size=12))
        
        st.plotly_chart(fig2)
        
        df['PosxMark'] = df['pos_markw'] - df['pos_atual']
        df['PosxOsc'] = (df['pos_markw']*df['pos_osc']/100) - df['pos_atual']
        
        st.markdown('---')
        st.subheader('Comparação entre posições')
        st.write(' Se > 0: Comprar, < 0: Vender')
        st.dataframe(df.set_index('Acao')[['PosxMark','PosxOsc']])
        
        ultimo_preco = df_prices.iloc[-1,:]
        st.write('Referencia dos Preços')
        st.write(ultimo_preco)
        
        
        
        
        
    
    # mercado
    # Tela que exibe informações de mercado das ações selecionadas: Gráficos de Preço e Volatilidade 
    # E parâmetros de carteira (Atual e de Markowitz): Vol, Retorno, Sharpe, Drawdown
    def mercado(self, name, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        
        dict_index = { #nomeIndex : Ticker  : yfinance
                      'USDBRL':'USDBRL=X',
                      'Gold':'GC=F',
                      'Silver': 'SI=F',
                      'Platinum': 'PL=F',
                      'Copper': 'HG=F',
                      'Paladium': 'PA=F',
                      'Crude Oil': 'CL=F',
                      'Heating Oil': 'HO=F',
                      'Natural Gas': 'NG=F',
                      'RBOB Gasoline': 'RB=F',
                      'Brent Crude Oil': 'BZ=F',
                      'Corn CME': 'ZC=F',
                      'Oat': 'ZO=F',
                      'KC HRW Wheat': 'KE=F',
                      'Rough Rice': 'ZR=F',
                      'Soybean Meal': 'ZM=F',
                      'Soybean Oil': 'ZL=F',
                      'Soybean': 'ZS=F',
                      'Feeder Cattle': 'GF=F',
                      'Lean Hogs': 'HE=F',
                      '	Live Cattle':'LE=F',
                      'Cocoa':'CC=F',
                      'Coffee ARA ICE': 'KC=F',
                      'Cotton': 'CT=F',
                      'Lumber': 'LBS=F',
                      'Orange Juice': 'OJ=F',
                      'Sugar11':'SB=F'
                        
            }
        
        dict_index_inv = dict(zip(dict_index.values(), dict_index.keys()))
        
        st.title('Informações de Mercado')
        st.markdown('---')
        
        m = Models()
        
        
        st.header(name)
        if name == "Ações" or name == "Fundos Imobiliários":
            lista_acoes = [x + '.SA' for x in list(df.ticker)]
        if name == 'Commodities' or name == 'Moedas':
            lista_acoes = [x for x in list(df.ticker)]
        if name == 'Indicadores':
            lista_acoes = ['^BVSP','^MERV','^GSPC','^DJI','NQ=F','^FTSE','^HSI','^N225', '^RUT','BOVA11.SA','SMAL11.SA','IFIX.SA','SPXI11.SA', 'XINA11.SA'] # A fazer
            
        #Puxando os preços
        df_prices = m.download_prices(lista_acoes, per_data, anos_cotacoes, datas_inicio, datas_fim)
        
        st.markdown('---')
        st.subheader('Cotações Intraday')
        st.markdown(date.today().strftime('%d/%m/%Y'))
        
        count=0
        cols = st.columns(3,gap='medium')
        
        
        df_info = pd.DataFrame({'Ativo': df_prices.columns})
        
        df_info['Ult. Valor'] = ''
        df_info['Var. %'] = ''
        
        fig = go.Figure()
        fig1 = go.Figure()
        fig1b = go.Figure()
        fig2 = go.Figure()
        fig3 = go.Figure()
        fig4 = go.Figure()
        
        retornos = df_prices.pct_change()
        retornos_ac = (1+retornos).cumprod()
        volatility = np.log(df_prices.ffill()).diff().ewm(com=32).std()
        
        for tick in df_prices.columns:
            #variação
            var = ((df_prices[tick].iloc[-1]/df_prices[tick].iloc[-2])-1)*100
            df_info['Ult. Valor'][count] = round(df_prices[tick].iloc[-1],2)
            df_info['Var. %'][count] = round(var,2)
            
            with cols[count%3]:
                if name == 'Commodities':
                    st.metric(dict_index_inv[tick], value=df_info['Ult. Valor'][count], delta=str(df_info['Var. %'][count])+'%')
                else:
                    st.metric(tick, value=df_info['Ult. Valor'][count], delta=str(df_info['Var. %'][count])+'%')
    
            count +=1
            
            cotacoes = df_prices[tick]
            
            fig.add_trace(go.Scatter(x=cotacoes.index, y=cotacoes, name=tick))
            fig1.add_trace(go.Scatter(x=retornos[tick].index, y=retornos[tick], name=tick))
            fig1b.add_trace(go.Box(y=retornos[tick], name=tick))
            fig2.add_trace(go.Scatter(x=retornos_ac[tick].index, y=retornos_ac[tick], name=tick))
            fig3.add_trace(go.Scatter(x=volatility[tick].index, y=volatility[tick]*100*np.sqrt(252), name=tick))
            fig4.add_trace(go.Box(y=volatility[tick]*100*np.sqrt(252), name=tick))
            
        st.markdown('---')
        st.subheader('Preços')
        st.plotly_chart(fig)
        
        st.markdown('---')
        st.subheader('Retornos')
        st.plotly_chart(fig1)
        st.plotly_chart(fig1b)
        #st.dataframe(describe.describe())
        
        st.markdown('---')
        st.subheader('Retornos Acumulados')
        st.plotly_chart(fig2)
        
        st.markdown('---')
        st.subheader(f'Volatility EWM 36 dias')
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)
        
        
        st.markdown('---')
        st.subheader(f'Preços Individuais')
        sel_indiv = st.selectbox('Selecione a Ação', df_prices.columns )
        
        if per_data == 'Períodos':
            candle_df = yf.download(sel_indiv, period=f'{anos_cotacoes}y')
        if per_data == 'Data':
            candle_df = yf.download(sel_indiv, start=datas_inicio , end=datas_fim)
        if per_data == 'Máx':
            candle_df = yf.download(sel_indiv, period='max')
        
        fig_i = go.Figure(data=[go.Candlestick(name=sel_indiv ,x=candle_df.index,
                                               open=candle_df['Open'],
                                               high=candle_df['High'],
                                               low=candle_df['Low'],
                                               close=candle_df['Close'])])
        fig_i.update_layout(title=sel_indiv, xaxis_rangeslider_visible=False)
        
        sma_short = 20
        sma_medium = 60
        sma_long = 100
        
        #add SMA
        sma_short_t = SMAIndicator(candle_df['Adj Close'], window=sma_short)
        candle_df['SMA_SHORT'] = sma_short_t.sma_indicator()
        sma_medium_t = SMAIndicator(candle_df['Adj Close'], window=sma_medium)
        candle_df['SMA_MID'] = sma_medium_t.sma_indicator()
        sma_long_t = SMAIndicator(candle_df['Adj Close'], window=sma_long)
        candle_df['SMA_LONG'] = sma_long_t.sma_indicator()
        
        fig_i.add_trace(go.Scatter(name=f'SMA_SHORT_{sma_short}d', x=candle_df.index, y=candle_df['SMA_SHORT']))
        fig_i.add_trace(go.Scatter(name=f'SMA_MID_{sma_medium}d', x=candle_df.index, y=candle_df['SMA_MID']))
        fig_i.add_trace(go.Scatter(name=f'SMA_LONG_{sma_long}d', x=candle_df.index, y=candle_df['SMA_LONG']))
        
        st.plotly_chart(fig_i)
            
            
        
        
        
    
    # modelos
    # Tela que exibe dados dos outputs para os modelos de Markwitz e Oscilador
    #
    def modelos(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Modelos')
        st.markdown('---')
        
        m = Models()
        df_prices = m.download_prices(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
        #st.write(df_prices)
        df_returns = m.returns(df_prices)
        #st.write(df_returns)
        
        
        modelo = st.radio('Escolha o modelo: ', ['Correlação','Oscilador','Markowitz'])
        
        mult_simb = st.multiselect('Escolha as ações: ', list(df_returns.columns.values), list(df_returns.columns.values))
        
        if modelo == 'Correlação':
            m.correlacao(df_prices.loc[:,mult_simb])
            return pd.DataFrame()
            
        if modelo == 'Oscilador':
            #df_osc = m.oscilador(df_prices.loc[:,mult_simb]).copy()
            df_osc = m.oscilador(df_prices).copy()
            df_send = df.copy().set_index('Acao')
            df_send.index = df_osc.index
            
            df_compare = df_send.copy()
            df_compare['pos_osc_pos'] = df_osc['pos_osc'].values
            
            st.write('Antes')
            st.write(df_send)
            
            df_send['pos_osc'] = df_osc[df_osc.columns].values
            st.write('Depois')
            st.write(df_send)
            
            st.write(df_compare[['pos_osc','pos_osc_pos']])
            
            return df_send
                
            
            
        if modelo == 'Markowitz':
            f_mark = df_prices.loc[:,mult_simb].copy()
            m.markowitz_inputs(f_mark,anos_cotacoes)
            return pd.DataFrame()
    
    #relatorio
    # Tela que exibe um relatório com a comparação dos retornos em períodos específicados
    #
    def relatorio(self):
        st.header('Relatório por Períodos')
    
    
        
        