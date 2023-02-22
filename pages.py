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
        
        
        
        
        
    
    # mercado
    # Tela que exibe informações de mercado das ações selecionadas: Gráficos de Preço e Volatilidade 
    # E parâmetros de carteira (Atual e de Markowitz): Vol, Retorno, Sharpe, Drawdown
    def mercado(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Informações de Mercado')
        st.markdown('---')
        st.dataframe(df)
        
        m = Models()
        df_prices = m.download_prices(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
        df_returns = m.returns(df_prices)
        st.write(df_returns)
        
    
    # modelos
    # Tela que exibe dados dos outputs para os modelos de Markwitz e Oscilador
    #
    def modelos(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Modelos')
        st.markdown('---')
        
        st.dataframe(df)
        
        m = Models()
        df_prices = m.download_prices(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
        df_returns = m.returns(df_prices)
        st.write(df_returns)
        
        
        modelo = st.radio('Escolha o modelo: ', ['Correlação','Oscilador','Markowitz'])
        
        #Ações
        lista_simb = inv.stocks.get_stocks(country='brazil')
        lista_ativos = []
        #st.write(tick + '.SA' for tick in lista_simb.symbol.unique())
        for tick in lista_simb.symbol.unique():
            lista_ativos.append(tick+'.SA')
            
        lista_ativos.append('USDBRL=X')
        lista_ativos.append('BTC-USD')
        
        st.write(lista_ativos[:,:])
        st.write(df_returns.columns.values)
        mult_simb = st.multiselect('Escolha as ações: ', lista_ativos[:,:], list(df_returns.columns.values))
        #mult_simb = st.multiselect('Escolha as ações: ', list(df_returns.columns.values), list(df_returns.columns.values))
        
        if modelo == 'Correlação':
            m.correlacao(df_prices)
            
        if modelo == 'Oscilador':
            m.oscilador()
            
        if modelo == 'Markowitz':
            m.markowitz()
    
    #relatorio
    # Tela que exibe um relatório com a comparação dos retornos em períodos específicados
    #
    def relatorio(self):
        st.header('Relatório por Períodos')
    
    
        
        