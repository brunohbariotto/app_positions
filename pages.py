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
        st.dataframe(df)
        
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
        st.write(df_prices)
        
        st.markdown('---')
        st.subheader('Cotações Intraday')
        st.markdown(date.today().strftime('%d/%m/%Y'))
        
        count=0
        cols = st.columns(3,gap='medium')
        
        
        df_info = pd.DataFrame({'Ativo': df_prices.columns})
        
        df_info['Ult. Valor'] = ''
        df_info['Var. %'] = ''
        
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
        
        st.write(df_info)
            
            
        
        
        
    
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
            
            st.write('Antes')
            st.write(df_send)
            
            df_send['pos_osc'] = df_osc[df_osc.columns].values
            st.write('Depois')
            st.write(df_send)
            
            return df_send
                
            
            
        if modelo == 'Markowitz':
            m.markowitz()
            return pd.DataFrame()
    
    #relatorio
    # Tela que exibe um relatório com a comparação dos retornos em períodos específicados
    #
    def relatorio(self):
        st.header('Relatório por Períodos')
    
    
        
        