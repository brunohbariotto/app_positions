# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:32:00 2023

@author: Dell inspiron
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf #poisson and binneg model
import statsmodels.api as sm #poisson and binneg family
from statsmodels.iolib.summary2 import summary_col
import plotly.graph_objects as go
import numpy as np
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity # Teste de esferacidade
from factor_analyzer import FactorAnalyzer
#from statstests.tests import overdisp


class Ml_models:
    def __init__(self, model, df, y_var, x_var):
        self.model = model
        self.df = df
        self.y_var = y_var
        self.x_var = x_var
        
    def choose_model(self):        
        
        if self.model == "Regressão Linear":
            st.markdown('---')
            st.header('**Modelos de Regressão**')
            
        elif self.model == "Contagem":
            st.markdown('---')
            st.header('**Modelos de Contagem**')
            
            st.markdown("**Modelos de Regressão**: **Poisson** e **Binomial Negativo**")

            st.markdown("**Y (Variável Dependente)**: Quantitativa com Valores Inteiros e não negativos")
            
            df_cont = self.df.copy()
            
            df_cont[self.x_var] = df_cont[self.x_var].astype('float')
            df_cont[self.y_var] = df_cont[self.y_var].astype('int64')
            formula = f'{self.x_var[0]} ~ ' + ' + '.join(self.x_var)
            
            #Description
            self.description_count(df_cont, self.y_var[0], self.x_var)
            
            #Poisson Model
            m_poi = self.poisson(df_cont, self.y_var[0], self.x_var, formula)
            
            
            st.subheader('Summary')
            st.write(m_poi.summary())
            
            
            st.write('Insira os Parâmetros')
            df_pred = pd.DataFrame(df_cont.iloc[-1]).T.loc[:, self.x_var]

            df_pred_input = st.experimental_data_editor(df_pred, num_rows="dynamic")
            
            df_pred_output = m_poi.predict(df_pred_input)
            
            st.subheader('Predict')
            st.write(df_pred_output)
            
            #Teste de Superdispersão
            st.markdown('---')
            st.header('Teste de Superdispersão de Cameron e Trivedi (1990)')
            
            st.latex(r''' T_{cam} = \frac{[(Y_{i} - \lambda_{poisson})^2 - Y_{i}]}{\lambda_{poisson}} ''')
            st.latex(r''' T_{cam} = \beta . \lambda_{poisson} ''')
            
            st.latex(r''' Cameron e Trivedi (1990): Superdispersão nos dados se \beta do modelo auxiliar T_{cam} sem intercepto é estatísticamente diferente de zero para dado nível de significancia''')
            
            st.latex( r'''Var \gg Média = \mu = \lambda_{poisson}''')
            
            
            df_cont['lambda_poisson'] = m_poi.fittedvalues
            
            df_cont['ystar'] = (((df_cont[self.y_var[0]]
                                  -df_cont['lambda_poisson'])**2)
                                -df_cont[self.y_var[0]])/df_cont['lambda_poisson']
            
            m_aux = smf.ols(formula='ystar ~ 0 + lambda_poisson',
                            data=df_cont).fit()
            
            st.write(m_aux.summary())
            st.write(f'p-value do lambda de poisson: {float(m_aux.pvalues[0])}')
            
            if float(m_aux.pvalues[0]) < 0.05:
                st.write('p-value do lambda_poisson < 0.05: Superdispersão.')
                st.header('Binomial Negativa')
                
            else:
                st.write('p-value do lambda_poisson > 0.05: Equidispersão.')
                st.write('O modelo de Poisson é suficiente!')
            
            
            
        elif self.model == "Regressão Logística":
            st.markdown('---')
            st.header('**Modelos de Regressão Logística**')
            
            
            
        elif self.model == "PCA":
            st.markdown('---')
            st.header('**Principal Component Analysis**')
            
            df_pca_inp = self.description_pca()
            
            self.correlation(df_pca_inp)
            
            if self.spheracity_bartlett(df_pca_inp):
                #Aplicação do PCA é adequada
                self.PCA(df_pca_inp)
                
            else:
                st.markdown('---')




        elif self.model == "Clusterização":
            st.markdown('---')
            st.header('**Modelos de Clusterização**')
            
            
    def description_pca(self):
        
        st.write('PCA é um algoritimo de aprendizagem de máquinas não-supervisionado, e portanto, não tem carater preditivo para observações que não estão presentes na amostra original')
        st.write('No entanto, é possível obter fatores resultantes das observações contidas no dataset')
        st.write('O PCA **transforma** as variaveis originais em outras variáveis ortogonais (entre si) por **redução de dimensionalidade** e consequentemente redução da variância')
        st.write('Assim, encontra-se as dimensões com a maior variabilidade explicativa dentre as variáveis originais ')
        st.write('Podemos aplicar o PCA para transformar os dados para posterior aplicação em algoritmos supervisionados, tais como Regressão ou Classificação')
        st.write('Com os valores transformados, podemos criar rankings dos fatores obtidos e ordenar para melhor visualização dos dados')
        st.write('Os Rankings são obtidos em cada fator são multiplicados pela % de variância compartilhada para a realização da ordenação')
        
        change_type = st.selectbox('PCA sobre ', ['Original', 'Variação %', 'Diferença'])
        
        df_pca = self.df.copy()
        
        if change_type == 'Original':
            df_pca_inp = df_pca.astype(float)
            
        elif change_type == 'Variação %':
            df_pca_inp = df_pca.pct_change.astype(float)
            
        elif change_type == 'Diferença':
            df_pca_inp = df_pca - df_pca.shift(1)
            df_pca_inp.dropna(inplace=True)
            df_pca_inp = df_pca_inp.astype(float)
            
        st.write(df_pca_inp)
        return df_pca_inp
            
    def description_count(self, df, y_var, x_var):
        st.markdown('---')
        st.header('**Description**')
        st.subheader('DataFrame')
        st.write(df)
        
        st.subheader('Info')
        st.write(df.info())
        
        st.subheader('Describe')
        st.write(df.describe())
        
        st.subheader('Histograma de Contagem')
        contagem = df[y_var].value_counts(dropna=False)
        percent = df[y_var].value_counts(dropna=False, normalize=True)
        st.write(pd.concat([contagem, percent], axis=1, keys=['count', '%'], sort=True))
        
        n_bins = st.number_input('Bins', min_value=1, value=20, step=1, key=13175)
        
        fig_hist_count = plt.figure(figsize=(20,15))
        sns.histplot(data=df, x=y_var, bins=n_bins, color='darkorchid')
        st.pyplot(fig_hist_count)
        
        st.subheader('Média e Variância de Y')
        st.write(pd.DataFrame({'Mean':[df[y_var].mean()],
                               'Variance':[df[y_var].var()]}))
        
        
        
    def PCA(self, df):
        st.markdown('---')
        st.header('PCA')
        fa = FactorAnalyzer()
        #Treinando o modelo com todos os fatores
        fa.fit(df)
        
        #Autovalores
        ev, v = fa.get_eigenvalues()
        st.subheader('Autovalores')
        st.write(ev)
        
        #Critério de Kaizer
        st.subheader('Critério de Kaiser ou Raizes Latentes')
        st.write('Apenas fatores cujos Autovalores > 1 devem ser considerados')
        st.write('Autovalores < 1 podem não ser tão representativos')
        st.write('Neste caso: {ev[ev > 1]}')

        #Retreinando o modelo com quantidade de fatores selecionados
        factors = st.number_input('Escolha o número de fatores que serão aplicados (default = Kaiser)', min_value=1, max_value=len(ev), value= len(ev[ev > 1]))
        fa.set_params(n_factors = factors, method = 'principal', rotation = None)
        fa.fit(df)
        
        #Variancia
        st.subheader('Variância')
        st.write('Os autovalores explicam a % de variancia compartilhada entre as variáveis originais que formaram cada fator')
        st.write('A Variância apresenta a proporção da variabilidade explicada individualmente (e cumulativa) de cada fator')

        eigen_fatores = fa.get_factor_variance()
        tabela_eigen = pd.DataFrame(eigen_fatores)
        tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
        tabela_eigen.index = ['Autovalores','Variância', 'Variância Acumulada']
        tabela_eigen = tabela_eigen.T

        st.write(tabela_eigen)
        
        #Fatores de carga
        st.subheader('Fatores de Carga')
        st.write('Representa a **Correlação de Pearson** entre as variáveis originais e os fatores criados')
        st.write('Mostra a importância de cada variável original na constituição de cada fator ortogonal criado ')
        st.write('Quanto maior o fator de carga, maior a influência daquela variável naquele fator')
        
        cargas_fatores = fa.loadings_
        tabela_cargas = pd.DataFrame(cargas_fatores)
        tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
        tabela_cargas.index = df.columns

        st.write(tabela_cargas)
        
        #Comunalidades
        st.subheader('Comunalidades')
        st.write('Mostra a perda de variância por variável após a exclusão dos fatores (por critério de raízes latentes por exemplo) ')
        st.write('Variância total compartilhada por cada variavel em todos os fatores extraídos e selecionados. ')

        comunalidades = fa.get_communalities()
        tabela_comunalidades = pd.DataFrame(comunalidades)
        tabela_comunalidades.columns = ['Communalities']
        tabela_comunalidades.index = df.columns

        st.write(tabela_comunalidades)
        
        #Transformações
        st.subheader('Transformações')
        
        predict_fatores= pd.DataFrame(fa.transform(df))
        predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]

        st.write(predict_fatores)
        notas = pd.concat([df.reset_index(drop=True), predict_fatores], axis=1)
        
        #Factor Scores
        st.subheader('Factor Scores')
        st.write('Parâmetros que relacionam os fatores criados com as variáveis originais, representados em um modelo linear')
        st.write('Provém dos autovalores e autovetores da matriz de correlação')
        st.write('Para um dataset com K variáveis, podemos ter até K fatores gerados')
        
        scores = fa.weights_
        tabela_scores = pd.DataFrame(scores)
        tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
        tabela_scores.index = df.columns

        st.write(tabela_scores)
        
        #Ranking
        st.subheader('Ranking')
        notas['Ranking'] = 0

        for index, item in enumerate(list(tabela_eigen.index)):
            variancia = tabela_eigen.loc[item]['Variância']
            notas['Ranking'] = notas['Ranking'] + notas[tabela_eigen.index[index]]*variancia

        notas.index = df.index
        st.write(notas)
        
        

        
    def spheracity_bartlett(self, df):
        st.markdown('---')

        st.header('**Adequação Global:** Bartellet Teste de Esferacidade')
        st.write('Compara a Matriz de significancia estatística da Correlação com a Matriz Identidade')
        st.write('É esperado que estas sejam **diferentes**, i.e, Correlações estatísticamente significativas para que o **PCA possa ser aplicado** ')
        
        st.write('**Teste de Hipótese:** p-value < 0.05 : Rejeição do H0 : PCA é aplicável!')
        
        bartlett, p_value = calculate_bartlett_sphericity(df)
        
        st.subheader(f'p-value : {round(p_value,5)}')
        
        if float(p_value) < 0.05:
            st.write('A Análise por componentes principais é aplicável, podemos prosseguir')
            return True
            
        else:
            st.write('A Matriz de correlação não se mostrou significativa estatísticamente, PCA não deve ser aplicado!')
            return False

        
    def correlation(self, df):
        st.markdown('---')
        st.subheader('Matriz de Correlação')
        
        df_corr = df.iloc[:,:].corr()
        
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
        
        
        
    def poisson(self, df, y_var, x_var, formula):
        st.markdown('---')
        st.header("**Poisson**")
        st.latex(r'''ln(\hat{Y}) = \alpha + \beta_{1}.X_{1} + ... + \beta_{k}.X_{k}''')
        st.markdown("**Probabilidade** de ocorrência de uma contagem m em dada exposição")
        st.latex(r'''p(Y_{i} = m) = \left(\frac{e^{-\lambda}.\lambda^{m}}{m!}\right)''')
        st.markdown("em que $\lambda$ é o número esperado de ocorrências ou taxa média estimada de incidências")
        st.latex( r'''Var \approx Média = \mu = \lambda_{poisson}''')
        
        m_poisson = smf.glm(formula=formula, 
                            data=df,
                            family = sm.families.Poisson()).fit()
        
        return m_poisson

        
            
        
