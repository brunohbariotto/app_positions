# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:34:08 2023

@author: Dell inspiron
"""

import streamlit as st

class Pages:
    def __init__(self, name):
        self.name == name
        
    # posicao
    # Tela que exibe a posição atual real bem como sua comparação com a posição de Markwitz e Oscilator
    #
    def posicao(self, df):
        st.header('Controle de Posição')
        st.dataframe(df)
    
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
    
    
        
        