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
import base64


loginSection = st.container()
logOutSection = st.container()
headerSection = st.container()

#Objeto que inicializa, lê e atualiza a planilha do google drive
gog = PlanGoogle()
#Objeto que inicializa e cria as páginas do menu principal
pg = Pages()

df_senha = gog.read_spreadsheet('Login')

def LoggedIn_Clicked(user_name, password):
    if user_name == str(df_senha.iloc[0,0]) and password == str(df_senha.iloc[0,1]):
        st.session_state['loggedIn'] = True
    else:
        st.session_state['loggedIn'] = False
        st.sidebar.error("Usuário/Senha Inválido")

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            st.sidebar.subheader('Login')
            user_name = st.sidebar.text_input(label = "", value="", placeholder="Usuário")
            password = st.sidebar.text_input(label = "", value="", placeholder="Senha",type="password")
            LoginButtonClicked = st.sidebar.button("Login", on_click=LoggedIn_Clicked, args=(user_name,password))
            
def LoggedOut_Clicked():
    st.session_state['loggedIn'] = False
            
def show_logout_page():
    loginSection.empty();
    with logOutSection:
        st.sidebar.button("Log out", key="logout", on_click=LoggedOut_Clicked)
                
                    
def show_main_page():
    st.sidebar.success('Logged In as {}'.format(str(df_senha.iloc[0,0])))

    lista_menu = ['Controle de Posição', 'Mercado','Modelos','Carteira', 'Machine Learning', 'Fundamentos']
    lista_tipo = ['Ações', 'Fundos Imob.']
    st.sidebar.subheader('Menu Principal')


    escolha = st.sidebar.radio('Escolha a Opção: ', lista_menu)

    st.sidebar.markdown('---')
    st.sidebar.subheader('Janela de datas')

    per_data = st.sidebar.radio('Por Período ou Datas', ['Períodos', 'Data', 'Máx'])

    anos_cotacoes = 1
    datas_inicio = datetime(2000,1,1)
    datas_fim = datetime.now()

    if per_data == 'Períodos':
    	anos_cotacoes = st.sidebar.slider('Quantos períodos (anos) de dados?', 1, 20, 8)
    elif per_data == 'Data':
    	datas_inicio = st.sidebar.date_input('Data de Inicio', value=datetime(2000,1,1) , min_value = datetime(2000,1,1), max_value=datetime.now())
    	datas_fim = st.sidebar.date_input('Data Final', value=datetime.now() , min_value = datetime(2000,1,1), max_value=datetime.now())


    # Escolha das páginas
    if escolha == 'Controle de Posição':
        df = gog.read_spreadsheet('positions_BrunoBariotto')
        df2 = gog.read_spreadsheet('2024')
        df2['Data'] = pd.to_datetime(df2['Data'])

        pg.posicao(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.subheader('Posição Começo de 2023')
        df_posi = df2.iloc[:,[1,2,4,5,6,9,12,13,14,15,16]]
        st.write(df_posi.drop_duplicates(subset=['Código'] ,keep='first').dropna())
        st.subheader('Última posição')
        st.write(df_posi.drop_duplicates(subset=['Código'], keep='last').dropna())
        
        st.write(df_posi[(df_posi['Operação'] == 'Venda')].groupby('Código')['Gain/Loss'].sum())

        
        st.header('Gain / Loss por mês')
       
        
        st.subheader('Ações')
        st.write('O pagamento é obrigatório quando efetuamos operações Day Trade com ganhos ou operações Swing Trade cuja soma das vendas em ações seja superior a R$ 20.000,00')
        st.write('Tributação: 15% sobre o ganho em operações swing trade e 20% sobre daytrade, DARF Sicalc é o 6015. OBS: Descontar 0.005% (1% para daytrade) retido na fonte (dedo duro)')
        st.write('Pagamento até o último dia útil do mês seguinte')
        st.write('Prejuízos em vendas inferiores a 20k mês podem ser usados na compensação e nunca prescrevem.')
        
        st.write('Vendas Ações')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'Ação')].groupby(by=['Mês'])['Total'].sum())
        st.write('Ganho / Prejuízo')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'Ação')].groupby(by=['Mês'])['Gain/Loss'].sum())
        
        st.subheader('ETFs')
        st.write('Imposto de 15% sobre o lucro em operações comuns e 20% day trade, e não há isenção de até 20 mil reais em vendas')
        st.write('Vendas ETFs')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'ETF')].groupby(by=['Mês'])['Total'].sum())
        st.write('Ganho / Prejuízo')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'ETF')].groupby(by=['Mês'])['Gain/Loss'].sum())
        
        st.subheader('FIIs')
        st.write('Imposto de 20% sobre o lucro em operações comuns e day trade, e não há isenção de até 20 mil reais em vendas')
        st.write('Vendas FIIs')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'FII')].groupby(by=['Mês'])['Total'].sum())
        st.write('Ganho / Prejuízo')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'FIIs')].groupby(by=['Mês'])['Gain/Loss'].sum())
        
        st.subheader('USDCoin')
        st.write('Isenção de IR para vendas mensais de até 35 mil reais. ')
        st.write('Vendas USDCoin')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'Cripto')].groupby(by=['Mês'])['Total'].sum())
        st.write('Ganho / Prejuízo')
        st.write(df_posi[(df_posi['Operação'] == 'Venda') & (df_posi['Type'] == 'Cripto')].groupby(by=['Mês'])['Gain/Loss'].sum())
        
        
    if escolha == 'Mercado':
        
        mkt = st.radio('Escolha o Mercado: ', ['Ações','Fundos Imobiliários','Commodities','Moedas', 'Indicadores'], horizontal=True)
        
        if mkt == 'Ações':
            df = gog.read_spreadsheet('acoes_BrunoBariotto')
            
        if mkt == 'Fundos Imobiliários':
            df = gog.read_spreadsheet('fundos_BrunoBariotto')
            
        if mkt == 'Commodities':
            df = gog.read_spreadsheet('commodities_BrunoBariotto')
            
        if mkt == 'Moedas':
            df = gog.read_spreadsheet('moedas_BrunoBariotto')
            
        if mkt == 'Indicadores':
            df = gog.read_spreadsheet('indicadores_BrunoBariotto')
            
        pg.mercado(mkt, df, per_data, anos_cotacoes, datas_inicio, datas_fim)
        

    if escolha == 'Modelos':
        escolha_tipo = st.radio('Escolha o Tipo: ', lista_tipo)
        if escolha_tipo == 'Ações':
            df = gog.read_spreadsheet('positions_BrunoBariotto')
        elif escolha_tipo == 'Fundos Imob.':
            df = gog.read_spreadsheet('fundos_BrunoBariotto')
            df.columns = ['Acao']
            df['Acao'] = df['Acao'] + '.SA'
            
        df_tosend = pg.modelos(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
        
        if len(df_tosend) != 0 and escolha_tipo=="Ações":
            df_tosend = df_tosend.reset_index()
            df_tosend.rename(columns={'index':'Acao'}, inplace=True)
        
            if st.button('Atualizar Oscilador?'):
                st.write('Enviando...')
                gog.update_spreadsheet('positions_BrunoBariotto', df_tosend)
                
    if escolha == 'Carteira':
        df = gog.read_spreadsheet('acoes_BrunoBariotto')
        pg.carteira(df, per_data, anos_cotacoes, datas_inicio, datas_fim)
        
    if escolha == 'Machine Learning':
        pg.machine()
        
    if escolha == 'Fundamentos':
        pg.fundamentos()
        
        

with headerSection:
    
    
    if 'loggedIn' not in st.session_state:
        st.session_state['loggedIn'] = False

        pg.curriculo()        
        # pdf_file = '/curriculum.pdf'
        
        # with open(pdf_file, "rb") as pdf_file:
        #     PDFbyte = pdf_file.read()
                    
        # st.download_button(label="Download Curriculum.pdf", key='3',
        #         data=PDFbyte,
        #         file_name="curriculum_brunobariotto.pdf",
        #         mime='application/octet-stream')
        
        
        # with open(pdf_file, "rb") as f:
        #     base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        # # Embedding PDF in HTML
        # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

        # # Displaying File
        # st.markdown(pdf_display, unsafe_allow_html=True)
        
        show_login_page()
    else:
        if st.session_state['loggedIn']:
            show_logout_page()
            show_main_page()
        else:
            st.header('Curriculum')
            show_login_page()
        


    
    
    
