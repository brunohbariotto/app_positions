# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:25:17 2023

@author: Dell inspiron
"""

import streamlit as st
import yfinance as yf
import pandas as pd
from pages import Pages
from plangoogle import PlanGoogle

#Objeto que inicializa, lê e atualiza a planilha do google drive
gog = PlanGoogle()
#Objeto que inicializa e cria as páginas do menu principal
pg = Pages()

lista_menu = ['Controle de Posição', 'Mercado','Modelos', 'Relatórios']
st.sidebar.subheader('Menu Principal')
escolha = st.sidebar.radio('Escolha a Opção: ', lista_menu)

# Escolha das páginas
if escolha == 'Controle de Posição':
    df = gog.read_spreadsheet('positions_BrunoBariotto')
    pg.posicao(df)
    
if escolha == 'Mercado':
    df = gog.read_spreadsheet('positions_BrunoBariotto')
    pg.mercado(df)

if escolha == 'Modelos':
    pg.modelos()
    
if escolha == 'Relatórios':
    pg.relatorio()
