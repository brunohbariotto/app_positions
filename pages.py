# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:34:08 2023

@author: Dell inspiron
"""

import streamlit as st
import plotly.graph_objects as go
from models import Models

class Pages:
    def __init__(self):
        pass
        
    # posicao
    # Tela que exibe a posição atual real bem como sua comparação com a posição de Markwitz e Oscilator
    #
    def posicao(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Controle de Posição')
        st.dataframe(df)
        
        m = Models()
        
        df_prices = m.download_prices(list(df.Acoes), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
        
        #Pizza da posição atual
        fig = go.Figure(data=[go.Pie(labels=df.Acao, values=df.pos_atual, textinfo='label+percent')])
        fig.layout.height = 700
        fig.layout.width = 700
        
        fig.update_layout(
            title={'text':'Posição Atual [% R$]'},
            font=dict(
                family="Courier New, monospace",
                size=10))
        
        st.plotly_chart(fig)
        
        #Comparação Posição atual x Markowitz x Oscilator
        fig2 = go.Figure(data=[
            go.Bar(name='Pos. Markowitz', x=df.Acao, y= df.pos_markw),
            go.Bar(name='Pos. Oscilator', x=df.Acao, y= df.pos_osc*df.pos_markw/100),
            go.Bar(name='Pos. Atual', x=df.Acao, y= df.pos_atual)
        ])
        fig2.layout.height = 700
        fig2.layout.width = 1000
        
        fig2.update_layout(
            title={'text':'Comparação entre as Posições: Markowitz, Oscilator e Atual'},
            xaxis_title='Ações',
            yaxis_title='Quantidade de Ações',
            font=dict(
                family="Courier New, monospace",
                size=18))
        
        st.plotly_chart(fig2)
        
        df['PosxMark'] = df['pos_markw'] - df['pos_atual']
        df['PosxOsc'] = (df['pos_markw']*df['pos_osc']/100) - df['pos_atual']
        
        st.dataframe(df.set_index('Acao')[['PosxMark','PosxOsc']])
        
        
        
        
        
    
    # mercado
    # Tela que exibe informações de mercado das ações selecionadas: Gráficos de Preço e Volatilidade 
    # E parâmetros de carteira (Atual e de Markowitz): Vol, Retorno, Sharpe, Drawdown
    def mercado(self, df):
        st.header('Informações de Mercado')
        st.dataframe(df)
        
    
    # modelos
    # Tela que exibe dados dos outputs para os modelos de Markwitz e Oscilador
    #
    def modelos(self):
        st.header('Modelos')
        
    
    #relatorio
    # Tela que exibe um relatório com a comparação dos retornos em períodos específicados
    #
    def relatorio(self):
        st.header('Relatório por Períodos')
    
    
        
        