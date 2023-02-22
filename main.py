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
from datetime import datetime, date


#Objeto que inicializa, lê e atualiza a planilha do google drive
gog = PlanGoogle()
#Objeto que inicializa e cria as páginas do menu principal
pg = Pages()

lista_menu = ['Controle de Posição', 'Mercado','Modelos', 'Relatórios']
st.sidebar.subheader('Menu Principal')
escolha = st.sidebar.radio('Escolha a Opção: ', lista_menu)

st.sidebar.markdown('---')
st.sidebar.subheader('Janela de datas')

per_data = st.sidebar.radio('Por Período ou Datas', ['Períodos', 'Data', 'Máx'])

anos_cotacoes = 1
datas_inicio = datetime(2000,1,1)
datas_fim = datetime.now()

if per_data == 'Períodos':
	anos_cotacoes = st.sidebar.slider('Quantos períodos (anos) de dados?', 1, 20)
elif per_data == 'Data':
	datas_inicio = st.sidebar.date_input('Data de Inicio', value=datetime(2000,1,1) , min_value = datetime(2000,1,1), max_value=datetime.now())
	datas_fim = st.sidebar.date_input('Data Final', value=datetime.now() , min_value = datetime(2000,1,1), max_value=datetime.now())


# Escolha das páginas
if escolha == 'Controle de Posição':
    df = gog.read_spreadsheet('positions_BrunoBariotto')
    pg.posicao(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
    
if escolha == 'Mercado':
    df = gog.read_spreadsheet('positions_BrunoBariotto')
    pg.mercado(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
    

if escolha == 'Modelos':
    df = gog.read_spreadsheet('positions_BrunoBariotto')
    df_tosend = pg.modelos(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
    
    st.write(df_tosend.reset_index())
    
    if st.button('Atualizar Oscilador?'):
        st.write('Enviando...')
        gog.update_spreadsheet('positions_BrunoBariotto', df_tosend)
    
if escolha == 'Relatórios':
    pg.relatorio()
