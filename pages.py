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
from ml_models import Ml_models
import requests


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
        st.write(df_prices)
        
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
        
        #markowitz_usd = 0
        #alterar aqui
        markowitz_usd = 30120*df[df['Acao'] == 'USDBRL=X']['pos_osc']/100
        #st.write('USD-BRL MarkxOsc')
        #st.write(markowitz_usd)
        #e aqui
        markowitz_btc = markowitz_usd*(1.62/(1.62+15.58))/1.5
        markowitz_spx = markowitz_usd*(15.56/(1.62+15.58))/1.5
        
        #st.write('BTC MarkxOsc')
        #st.write(((markowitz_btc)/(df_prices['HASH11.SA'].iloc[-1])).iloc[0])
        #st.write('SPX MarkxOsc')
        #st.write(((markowitz_spx)/(df_prices['SPXI11.SA'].iloc[-1])).iloc[0])
        
        #st.write(df.loc[df['Acao']=='HASH11.SA','pos_markw'].iloc[0])
        #st.write(df.loc[df['Acao']=='SPXI11.SA','pos_markw'].iloc[0])
        
        #st.write()
        
        df.loc[(df['Acao']=='HASH11.SA'),['pos_markw']] = df.loc[df['Acao']=='HASH11.SA','pos_markw'].iloc[0] + ((markowitz_btc)/(df_prices['HASH11.SA'].iloc[-1])).iloc[0]
        df.loc[(df['Acao']=='IVVB11.SA'),['pos_markw']] = df.loc[df['Acao']=='IVVB11.SA','pos_markw'].iloc[0] + ((markowitz_spx)/(df_prices['IVVB11.SA'].iloc[-1])).iloc[0]
        
        
        df['pos_oscxmark'] = df.pos_osc*df.pos_markw/100
        
        #st.write(df_prices)
        
        #st.write(df)
        
        #Comparação Posição atual x Markowitz x Oscilator
        st.markdown('---')
        st.subheader('Comparação entre as Posições: Markowitz, Oscilator e Atual')
        fig2 = go.Figure(data=[
            go.Bar(name='Pos. Markowitz', x=df.Acao, y= df.pos_markw),
            go.Bar(name='Pos. Oscilator', x=df.Acao, y= df.pos_oscxmark),
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
        df['PosxOsc'] = (df['pos_oscxmark']) - df['pos_atual']
        
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
        
        st.write(df_prices)
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
            
            
        
        
    def carteira(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Carteira')
        st.markdown('---')
        
        df['ticker'] = df.ticker.apply(lambda l: l+".SA")
        
        m = Models()
        df_prices = m.download_prices(list(df.ticker), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
        df_returns = m.returns(df_prices)
        
        df_info = pd.DataFrame(index=df.ticker)
        
        st.subheader('Retorno Anual Individual - Base 246 dias úteis [%]')
        df_info['Ret_Anual'] = df_returns.mean() * 246 * 100
        st.write(df_info['Ret_Anual'])
        
        st.write('Last Prices')
        df_info['Last_price'] = df_prices.iloc[-1,:]
        st.write(df_info['Last_price'])
        
        
        st.write('% Alocação por ativo')
        df_info['%_aloc'] = (df_prices.iloc[-1]* df.qtdd)/(df.qtdd.sum())
        st.write(df_info['%_aloc'])
        
        st.write('Retorno Anual Carteira - Base 246 dias úteis [%]')
        st.write(np.dot(df_info['Ret_Anual'] , np.array(list(df_info['%_aloc']))))
    
    # modelos
    # Tela que exibe dados dos outputs para os modelos de Markwitz e Oscilador
    #
    def modelos(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Modelos')
        st.markdown('---')
        
        m = Models()
        df_prices = m.download_prices(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
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
            

            
            df_compare['pos_osc_pos'] = df_osc[df_osc.columns]#.values
            
            st.write('Antes')
            st.write(df_send)
            
            df_send['pos_osc'] = df_osc[df_osc.columns]#.values

            
            st.write(df_compare[['pos_osc','pos_osc_pos']])
            
            return df_send
                
            
            
        if modelo == 'Markowitz':
            f_mark = df_prices.loc[:,mult_simb].copy()
            m.markowitz_inputs(f_mark,anos_cotacoes)
            return pd.DataFrame()
        
    
    def curriculo(self):
        #st.set_page_config(layout='wide')

        col1T, col2T = st.columns(2)

        with col1T:
        
            st.title('Bruno Bariotto')
        
            st.markdown(':house: Campinas-SP, Brazil')
        
        st.markdown('---')

        col1P, col2P, col3P, col4P, col5P = st.columns(5)
        
        with col1P:
        
            st.markdown(':iphone: +55 (19) 98175-8460')
        
        with col2P:
        
            st.markdown(':email: brunohbariotto@gmail.com')
        
        with col3P:
        
            st.markdown(':earth_americas: https://brunobariotto.streamlit.app/')
        
        with col4P:
        
            st.markdown('www.linkedin.com/in/bruno-henrique-bariotto')
        
        with col5P:
        
            st.markdown('https://github.com/brunohbariotto')   
        
         
        
        st.markdown('---')
        
        st.markdown(
        
        """
        
        - Solid Knowledge in Derivatives, Commodities market and Exotic products;
        
        - Experience in Price, Trade, Hedge and book Exotic and hybrid structures, using Portfolio and Risk Management to centralize risk of exotic options / non-linear book
        
        - Dynamically hedge in books, to manage the position and preserve the P&L;
        
        - Multi-Asset class exposure (Ags, Softs, Energy, FX);
        
        - Market-Making on low liquidity markets, offering prices in B3 and Matif Corn Options for brokers and sales;
        
        - Execute Prop Trading Strategies on volatility, Future calendar spread and Intermarket spread;
        
        - Structuring of controls, reports, quantitative strategies and pricing tools to frame and have the best view on systematic, market, liquidity risks.

        """
        
        )

        st.subheader('Skills:')
        
        st.markdown('Programming Languages: C, C++, Python, R, JAVA, Matlab, VHDL, VBA, Assembly')
        
        st.markdown('Trading: Finance, Derivatives, Risk Management, Pricing, Proprietary, Futures, Options and Greeks')
        
         
        
        st.markdown('---')
        
        st.header('Education')
        

        st.markdown('''
        
        <style>
        
        .katex-html {
        
            text-align: left;
        
        }
        
        </style>''',
        
        unsafe_allow_html=True
        
        )
        
        
        st.latex(r'''
        
            \bullet\textbf{ UNICAMP - University of Campinas [Graduated], \text{2014-2019}}
        
        ''')
        
        st.latex(r'''
        
            \text{ B. Sc. In Control and Automation Engineering}
        
        ''')
        
        st.latex(r'''
        
            \bullet\textbf{EPAT in QuantInsti \text{2021-2022}}
        
        ''')
        
        st.latex(r'''
        
            \text{ Executive Programme in Algorithmic Trading}
        
        ''')
        
        st.latex(r'''
        
            \bullet\textbf{ USP/ESALQ - University of São Paulo, \text{2022-2024}}
        
        ''')
        
        st.latex(r'''
        
            \text{ Master of Business Adminstration - MBA In Data Science and Analytics}
        
        ''')

        
        st.markdown('---')
        
        st.header('Experience')
        
        st.latex(r'''
        
            \bullet\textbf{ Senior OTC Derivatives Trader at EDF Man Capital Markets / HedgePoint Global Markets, \text{Jan 2019 - now}}
        
        ''')
        
        st.markdown(
        
        """
        
        - During the entire experience, in charge of Ags and Soft books such as Corn, Wheat, Coffee, Sugar, Cotton, etc.
        
        - Trading, Dinamic Hedge for OTC Structures (Strips, Barrier, Compo, Spread Option, etc.)
        
        - MM on iliquid markets (B3 and Matif Options)
        
        - Statistical, regression and quantitative models to find good opportunities for a proprietary trading book and better hedge decisions.

        """
        
        )
    
    #relatorio
    # Tela que exibe um relatório com a comparação dos retornos em períodos específicados
    #
    def machine(self):
        st.header('Machine Learning Models')
        st.markdown('---')
        
        tipo = st.radio('Escolha o tipo: ', ['Supervised Learning', 'Unsupervised Learning'], key=13170)
        
        if tipo == 'Supervised Learning':
            st.subheader('Algoritmos Supervisionados')
            
            col1, col2 = st.columns(2)
            
            with col1:
                algo = st.selectbox('Selecione o algoritmo', ('Regressão', 'Classificação'))
                
            with col2:
                if algo == 'Regressão':
                    modelo_ml = st.radio('Escolha o Modelo: ', ['Regressão Linear', 'Contagem'], key=13171)
                elif algo == 'Classificação':
                    modelo_ml = st.radio('Escolha o Modelo: ', ['Regressão Logística'], key=13172)
                    
        elif tipo == 'Unsupervised Learning':
            st.subheader('Algoritmos Não-Supervisionados')
            
            modelo_ml = st.radio('Escolha o Modelo: ', ['PCA', 'Clusterização'], key=13173)
            
        
        st.write('Insira a base de dados')
        
        input_type = st.selectbox('Escolha o tipo de input', ['Manual', 'Arquivo'])
        
        if input_type == 'Manual':
            n_cols = st.number_input('Selecione o número de colunas', min_value=1, value=1, step=1)
            
            cols = [f'col_{n}' for n in range(n_cols)]
            
            df_m = pd.DataFrame(
                np.array([['Column_Name']*n_cols,[1]*n_cols]),
                columns = cols
                )
    
            df_input = st.experimental_data_editor(df_m, num_rows="dynamic")
            
            df_input.columns = df_input.iloc[0,:]
            df_input = df_input.iloc[1:]
            
        
        elif input_type == 'Arquivo':
            uploaded_file = st.file_uploader('Escolha um arquivo')
            if uploaded_file is not None:
                df_input = pd.csv(uploaded_file)
                
                
        if tipo == 'Supervised Learning':
            try:
                st.write(df_input)
                
                list_xy = list()
                for c in df_input.columns:
                    list_xy.append([c, False, True])
                    
                df_xy_in = pd.DataFrame(np.array(list_xy),
                             columns = ['Variable', 'is_Y', 'is_X']
                             )
                
                df_xy_in['Variable'] = df_xy_in['Variable'].astype('string')
                df_xy_in['is_X'] = df_xy_in['is_X'].astype('bool')
                df_xy_in['is_Y'] = df_xy_in['is_Y'].astype('bool')
                    
                x_y_df = st.experimental_data_editor(
                    df_xy_in
                    )
    
                x_var = list(x_y_df[x_y_df.is_X == True]['Variable'].values)
                y_var = list(x_y_df[x_y_df.is_Y == True]['Variable'].values)
                
                st.write(f'Variável Dependente Y: {y_var}')
                st.write(f'Variáveis Independentes X: {x_var}')
                
                ml = Ml_models(modelo_ml, df_input, y_var, x_var )
                ml.choose_model()
            except:
                st.write('Insira os dados no dataframe acima')
                
        elif tipo == 'Unsupervised Learning':
            ml = Ml_models(modelo_ml, df_input, [], [])
            ml.choose_model()
            
    def fundamentos(self):
        st.header('Fundamentos')
        st.markdown('---')
        
        tipo = st.radio('Escolha o tipo: ', ['Fundos Imobiliários', 'Ações'], key=131554)
        
        if tipo == 'Fundos Imobiliários':
            st.subheader('Fundos Imobiliários')
            
            #scrapping the data
            url="https://www.fundsexplorer.com.br/ranking"
            response = requests.get(url)
            
            if response.status_code == 200:
                df = pd.read_html(response.content, encoding='utf-8')[0]
            else:
                st.markdown('Não foi possível fazer o WebScrapping dos dados!')
                return
            
            st.write(df)
            
            df.sort_values('Código do fundo', inplace=True)

            
            #removendo setores nulos
            df.drop(df[df['Setor'].isna()].index, inplace=True)

            setores = df['Setor'].unique()
            
            #Transformando dados categóricos
            categorical_columns = ['Código do fundo', 'Setor']
            df[categorical_columns] = df[categorical_columns].astype('category')
            
            #Transformando dados float: todas exceto código e setor
            col_floats = list(df.iloc[:,2:-1].columns)
            #preenchendo nan pra zero
            df[col_floats] = df[col_floats].fillna(value=0)
            #Separando Patrim. Liquido para replaces
            col_floats.remove('Patrimônio Líq.')
            df['PatrimônioLíq.'] = df[['Patrimônio Líq.']].applymap(lambda x: 
                                                                   str(x).replace('R$','').replace('.','').replace('%','').replace(',','.'))
                
            #df['Patrimônio Líq.'] = df['Patrimônio Líq.'].astype('float')

            df[col_floats] = df[col_floats].applymap(lambda x: 
                                                                   str(x).replace('R$','').replace('.0','').replace('.','').replace('%','').replace(',','.'))
              
            df[col_floats] = df[col_floats].astype('float')
            
            df['P/VPA'] = df['P/VPA']/100
            
            st.markdown('---')
            st.subheader('Setor:')
            escolha2 = st.radio('Escolha o Setor para analisar: ', setores, horizontal=True)
            st.markdown(escolha2)
            
            list_funds = list(df[df['Setor'] == escolha2].sort_values('Liquidez Diária', ascending=False)['Código do fundo'].iloc[:10].values)
            my_new_list = [x + '.SA' for x in list_funds]
            
            
            dict_top = {list_funds[i]: my_new_list[i] for i in range(len(my_new_list))}
            
            def find_fundos(df, media_setor, setor, metricas):
                metric = []
                equal = []
                value = []
                count_key=9958
                
                st.subheader(f'Encontre Fundos do Setor {setor}:')
                st.write("Entre com as condições: ")
                for i in metricas:
                    st.markdown(f'**{i}**')
                    st.write(count_key)
                    maior_menor = st.selectbox('', ('>', '<'), key=count_key)
                    
                    count_key = count_key + 100
                    metric_value = st.number_input('', 
                                                   value= round(df.groupby('Setor').agg(['mean','std']).loc[setor].loc[i].iloc[0],2), key = count_key+10)
                    
                    st.markdown(f"_Condição procurada: {i} {maior_menor} {metric_value}_ ")
                    metric.append(str(i).replace(' ','').replace('.','').replace('(','').replace(')','').replace('/',''))
                    equal.append(maior_menor)
                    value.append(metric_value)
                
                query = ' and '.join(['{}{}{}'.format(i,j,k) for i, j, k in zip(metric, equal, value)])
                    
                st.markdown(f'**Os seguintes fundos do setor {setor} foram encontrados para as condições:**')
                st.markdown(f"{query}")
                
                df.columns = df.columns.str.replace(' ','')
                df.columns = df.columns.str.replace('.','')
                df.columns = df.columns.str.replace('(','')
                df.columns = df.columns.str.replace(')','')
                df.columns = df.columns.str.replace('/','')
                    
                st.dataframe(df.query(query))
            
            def stats_fundos(df, setor, metricas, compare='>'):
                
                
                df_setor = df[df['Setor'] == setor]
                
                
                
                for i in metricas:
                    media_setor = round(df.groupby('Setor').agg(['mean','std']).loc[setor].loc[i].iloc[0],2)
                    std_setor = round(df.groupby('Setor').agg(['mean','std']).loc[setor].loc[i].iloc[1],2)
                    
                    st.subheader(i)
                    
                    st.write(f'A Média de {i} do Setor {setor} é de {media_setor}')
                    st.write(f'O desvio de {i} do Setor {setor} é de {std_setor}')
                    
                    #st.dataframe(df_setor[df_setor[i] >= media_setor] if compare == '>' else df_setor[df_setor[i] <= media_setor])
                    
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_setor[i],
                        y=df_setor['Código do fundo'],
                        orientation='h'))
                    fig.add_vline(x=media_setor, line_dash="dot", annotation_text="mean", line_color="red")
                    fig.update_layout(width=750, height=600)
                    
                    st.plotly_chart(fig)
                    
                find_fundos(df_setor, media_setor, setor, metricas)
            
            
            def fundamentos_fundos(df, setores, categorical_columns,escolha2):
                okplot = False
                st.title('Fundamentos')
                st.markdown('---')

                st.subheader('Descrição Geral')
                st.dataframe(df.describe())
                
                st.subheader('Média e Desvio-Padrão por Setor')
                st.dataframe(df.groupby('Setor').agg(['mean','std']))

                st.subheader('Indicadores:')
                indicadores = categorical_columns
                multi = st.multiselect("Escolha os indicadores: ", list(df.columns[2:]))
                #st.markdown(multi)
                for i in multi:
                    indicadores.append(i)
                    okplot = True
                    
                
                df_ind = df.loc[:,indicadores]

                if okplot:
                    stats_fundos(df_ind, setor=escolha2, metricas=multi, compare='>')
                    
                
            fundamentos_fundos(df,setores, categorical_columns,escolha2)
                
            

        
            
        
            

        
    
    
        
        