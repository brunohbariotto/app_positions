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
import io
from datetime import datetime, date
import numpy as np
from ta.trend import SMAIndicator
import yfinance as yf
from ml_models import Ml_models
import requests
import unicodedata


class Pages:
    def __init__(self):
        pass

    def _normalize_positions_df(self, df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        def _norm(s):
            s = str(s)
            s = s.replace('\ufeff', '')
            s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
            return s.strip().lower().replace(" ", "").replace("_", "")

        col_map = {_norm(c): c for c in df.columns}

        def _rename_if_missing(target, aliases):
            if target in df.columns:
                return
            for a in aliases:
                na = _norm(a)
                if na in col_map:
                    df.rename(columns={col_map[na]: target}, inplace=True)
                    return

        _rename_if_missing('Acao', ['Ação', 'Acao', 'Ativo', 'Ativos', 'Ticker', 'Código', 'Codigo', 'Cod', 'Asset'])
        _rename_if_missing('pos_atual', ['Posição Atual', 'Pos Atual', 'PosicaoAtual', 'PosAtual', 'Quantidade', 'Qtde', 'Qtd', 'Qtdade', 'Pos'])
        _rename_if_missing('pos_markw', ['pos_markw', 'pos_markowitz', 'Markowitz', 'PosMark', 'Pos_Mark'])
        _rename_if_missing('pos_osc', ['pos_osc', 'osc', 'PosOsc', 'Pos_Osc', 'Oscilador'])

        return df

    def _download_fred_series(self, series_id, column_name, start_date=None, end_date=None):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            st.warning(f"Falha ao buscar FRED ({series_id}): {exc}")
            return pd.DataFrame(columns=[column_name])

        if df.empty:
            return pd.DataFrame(columns=[column_name])

        date_col = 'DATE' if 'DATE' in df.columns else df.columns[0]
        value_col = series_id if series_id in df.columns else df.columns[-1]

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]

        df = df.rename(columns={value_col: column_name})[[column_name]]
        return df

    def alocacao_por_ativo(self, df_asset, df_positions, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.title('Alocacao por Classes de Ativos')
        st.markdown('---')
        df_asset = self._normalize_positions_df(df_asset)
        required_cols = {'Acao', 'pos_atual', 'pos_markw'}
        missing = required_cols.difference(set(df_asset.columns))
        if missing:
            st.warning(f"Colunas obrigatorias ausentes na aba Alocacao_Asset: {', '.join(sorted(missing))}")
            st.write('Colunas encontradas:', list(df_asset.columns))
            return
        for col in ['pos_atual', 'pos_markw']:
            df_asset[col] = pd.to_numeric(df_asset[col], errors='coerce').fillna(0.0)
        df_asset['Acao'] = df_asset['Acao'].astype(str).str.strip()
        if per_data == 'Períodos':
            start_date = pd.Timestamp(datetime.now()) - pd.DateOffset(years=anos_cotacoes)
            end_date = pd.Timestamp(datetime.now())
        elif per_data == 'Data':
            start_date = pd.Timestamp(datas_inicio)
            end_date = pd.Timestamp(datas_fim)
        else:
            start_date = None
            end_date = None
        m = Models()
        ticker_map = {
            '^BVSP.SA': '^BVSP',
            'SMAL11.SA': 'SMAL11.SA',
            'USDBRL=X': 'USDBRL=X',
            'IVVB11.SA': 'IVVB11.SA',
            'GOLD11.SA': 'GOLD11.SA',
            'HASH11.SA': 'HASH11.SA',
        }
        df_yahoo = m.download_prices_novo(list(dict.fromkeys(ticker_map.values())), per_data, anos_cotacoes, datas_inicio, datas_fim)
        if not df_yahoo.empty:
            inverse_map = {yf_ticker: label for label, yf_ticker in ticker_map.items()}
            df_yahoo = df_yahoo.rename(columns=inverse_map)
            for col in ticker_map.keys():
                if col not in df_yahoo.columns:
                    df_yahoo[col] = np.nan
            df_yahoo = df_yahoo[list(ticker_map.keys())]
        df_fred = self._download_fred_series('INTGSTBRM193N', 'Taxa_Juros_Brasil', start_date=start_date, end_date=end_date)
        df_consolidado = pd.concat([df_fred, df_yahoo], axis=1, join='outer').sort_index()
        if df_consolidado.empty:
            st.warning('Nao foi possivel consolidar os dados de FRED e Yahoo para o periodo selecionado.')
            return
        daily_index = pd.date_range(df_consolidado.index.min(), df_consolidado.index.max(), freq='D')
        df_consolidado = df_consolidado.reindex(daily_index).ffill()
        first_dates = []
        for col in ticker_map.keys():
            if col in df_consolidado.columns:
                first_valid = df_consolidado[col].first_valid_index()
                if first_valid is not None:
                    first_dates.append(first_valid)
        if first_dates:
            data_inicio_exibicao = max(first_dates)
            df_consolidado = df_consolidado[df_consolidado.index >= data_inicio_exibicao]
        df_consolidado.index.name = 'Data'
        taxa_ref_series = pd.to_numeric(df_consolidado.get('Taxa_Juros_Brasil'), errors='coerce')
        if isinstance(taxa_ref_series, pd.Series) and taxa_ref_series.notna().any():
            taxa_juros_brasil_ult = float(taxa_ref_series.dropna().iloc[-1])
        else:
            taxa_juros_brasil_ult = 2.0
        selic_aa_markowitz = taxa_juros_brasil_ult / 100.0
        tab_classes, tab_ibov = st.tabs(['Classe de Ativos', 'Acoes IBOV'])
        df_asset_plot = pd.DataFrame()
        with tab_classes:
            st.caption('Consolidacao diaria sem filtro de business day para preservar publicacoes do FRED em fins de semana.')
            if first_dates:
                st.caption(f'Exibicao iniciada em {data_inicio_exibicao:%Y-%m-%d}, data do ultimo ativo com historico disponivel.')
            st.dataframe(df_consolidado, use_container_width=True)
            st.markdown('---')
            st.subheader('Oscilador')
            df_pos_osc = pd.DataFrame(columns=['Ticker', 'pos_osc'])
            ativos_osc = [a for a in list(df_asset['Acao']) if a in df_consolidado.columns]
            if not ativos_osc:
                st.warning('Nenhum ticker da aba Alocacao_Asset foi encontrado na consolidacao de precos.')
            else:
                df_prices_osc = df_consolidado[ativos_osc].copy().ffill().dropna(how='all')
                df_pos_osc = m.oscilador(df_prices_osc).copy()
                if not df_pos_osc.empty:
                    if 'pos_osc' not in df_pos_osc.columns:
                        df_pos_osc.columns = ['pos_osc']
                    df_pos_osc = df_pos_osc.reindex(ativos_osc).rename_axis('Ticker').reset_index()
                    df_pos_osc['pos_osc'] = pd.to_numeric(df_pos_osc['pos_osc'], errors='coerce')
                    st.subheader('pos_osc por Ticker')
                    st.dataframe(df_pos_osc[['Ticker', 'pos_osc']], use_container_width=True)
            st.markdown('---')
            st.subheader('Comparacao entre as Posicoes: Markowitz, Oscilator e Atual')
            df_asset_plot = df_asset.copy()
            if 'pos_osc' in df_asset_plot.columns:
                df_asset_plot = df_asset_plot.drop(columns=['pos_osc'])
            if not df_pos_osc.empty:
                df_asset_plot = df_asset_plot.merge(df_pos_osc.rename(columns={'Ticker': 'Acao'}), on='Acao', how='left')
            else:
                df_asset_plot['pos_osc'] = np.nan
            df_asset_plot['pos_osc'] = pd.to_numeric(df_asset_plot['pos_osc'], errors='coerce').fillna(0.0)
            df_asset_plot['pos_oscxmark'] = df_asset_plot['pos_osc'] * df_asset_plot['pos_markw'] / 100
            fig = go.Figure(data=[
                go.Bar(name='Pos. Markowitz', x=df_asset_plot['Acao'], y=df_asset_plot['pos_markw']),
                go.Bar(name='Pos. Oscilator', x=df_asset_plot['Acao'], y=df_asset_plot['pos_oscxmark']),
                go.Bar(name='Pos. Atual', x=df_asset_plot['Acao'], y=df_asset_plot['pos_atual'])
            ])
            fig.update_layout(title={'text': 'Comparacao entre as Posicoes: Markowitz, Oscilator e Atual'}, xaxis_title='Ativos', yaxis_title='Quantidade', font=dict(family='Courier New, monospace', size=12), height=700, width=800)
            st.plotly_chart(fig)
            st.markdown('---')
            st.subheader('Distribuicao por Ativo - Pizzas Comparativas')
            total_markw = float(df_asset_plot['pos_markw'].sum())
            total_osc = float(df_asset_plot['pos_oscxmark'].sum())
            total_atual = float(df_asset_plot['pos_atual'].sum())
            for title, values in [(f'Posicao Markowitz - Soma Total: {total_markw:,.2f}', df_asset_plot['pos_markw']), (f'Posicao Oscilator - Soma Total: {total_osc:,.2f}', df_asset_plot['pos_oscxmark']), (f'Posicao Atual - Soma Total: {total_atual:,.2f}', df_asset_plot['pos_atual'])]:
                fig_p = go.Figure(data=[go.Pie(labels=df_asset_plot['Acao'], values=values, textinfo='label+value+percent')])
                fig_p.update_layout(title={'text': title}, font=dict(family='Courier New, monospace', size=12), height=680, legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5))
                st.plotly_chart(fig_p, use_container_width=True)
            csv_data = df_consolidado.reset_index().to_csv(index=False).encode('utf-8-sig')
            st.download_button('Baixar dados consolidados (CSV)', data=csv_data, file_name='alocacao_por_ativo_consolidado.csv', mime='text/csv')
            st.markdown('---')
            st.subheader('Tabela Comparativa por Ticker')
            df_comp = pd.DataFrame({'Ticker': df_asset_plot['Acao'], 'Posicao Atual': pd.to_numeric(df_asset_plot['pos_atual'], errors='coerce').fillna(0.0), 'Posicao Markowitz': pd.to_numeric(df_asset_plot['pos_markw'], errors='coerce').fillna(0.0), 'Posicao Oscilator': pd.to_numeric(df_asset_plot['pos_oscxmark'], errors='coerce').fillna(0.0)})
            df_comp['Posicao Atual - Markowitz'] = df_comp['Posicao Atual'] - df_comp['Posicao Markowitz']
            df_comp['Posicao Final - Posicao Oscilator'] = df_comp['Posicao Atual'] - df_comp['Posicao Oscilator']
            total_row = {'Ticker': 'TOTAL'}
            for col in df_comp.columns:
                if col != 'Ticker':
                    total_row[col] = pd.to_numeric(df_comp[col], errors='coerce').fillna(0.0).sum()
            df_comp = pd.concat([df_comp, pd.DataFrame([total_row])], ignore_index=True)
            st.dataframe(df_comp, use_container_width=True)
            valor_osc_bvsp = 0.0
            row_bvsp = df_asset_plot[df_asset_plot['Acao'] == '^BVSP.SA']
            if not row_bvsp.empty:
                valor_osc_bvsp = float(pd.to_numeric(row_bvsp['pos_oscxmark'], errors='coerce').fillna(0.0).iloc[0])
            st.session_state['aloc_ibov_valor'] = valor_osc_bvsp * 1000
        with tab_ibov:
            st.markdown('---')
            st.subheader('Alocacao por Acoes')
            valor_alocar_acoes = float(st.session_state.get('aloc_ibov_valor', 0.0))
            st.metric('Valor para alocar em Acoes (Pos. Oscilator de ^BVSP.SA)', f'R$ {valor_alocar_acoes:,.2f}')
            df_positions = self._normalize_positions_df(df_positions)
            if 'Acao' not in df_positions.columns:
                st.warning("Coluna obrigatoria nao encontrada em positions_BrunoBariotto: 'Acao'.")
                return
            df_positions['Acao'] = df_positions['Acao'].astype(str).str.strip()
            ativos_mapeados = set(df_asset['Acao'].astype(str).str.strip())
            ativos_acoes = sorted({t for t in df_positions['Acao'] if t and t.lower() != 'nan' and t not in ativos_mapeados})
            if not ativos_acoes:
                st.info('Nenhum novo ticker encontrado em positions_BrunoBariotto apos excluir os ativos mapeados.')
                return
            st.caption(f'{len(ativos_acoes)} tickers para analise de acoes (fora da tabela Alocacao_Asset).')
            df_prices_acoes = m.download_prices_novo(ativos_acoes, per_data, anos_cotacoes, datas_inicio, datas_fim)
            if df_prices_acoes.empty:
                st.warning('Nao foi possivel obter precos no Yahoo Finance para os tickers de acoes selecionados.')
                return
            st.subheader('Precos das Acoes (Yahoo Finance)')
            st.dataframe(df_prices_acoes, use_container_width=True)
            ultimos_precos_tela = pd.to_numeric(df_prices_acoes.ffill().iloc[-1], errors='coerce')
            st.markdown('---')
            st.subheader('Alocacao Markowitz para Acoes')
            valor_total_markowitz = valor_alocar_acoes
            st.metric('Valor Total para Markowitz (Pos. Oscilator ^BVSP.SA x 1000)', f'R$ {valor_total_markowitz:,.2f}')
            if valor_total_markowitz <= 0:
                st.warning('Valor total para Markowitz menor ou igual a zero. Ajuste a posicao de ^BVSP.SA.')
                return
            df_prices_mk = df_prices_acoes.copy().ffill().dropna(how='all')
            if df_prices_mk.empty:
                st.warning('Sem dados validos para executar Markowitz.')
                return
            if len(df_prices_mk.columns) == 1:
                ticker_unico = df_prices_mk.columns[0]
                ultimo_preco = float(pd.to_numeric(ultimos_precos_tela.get(ticker_unico), errors='coerce'))
                if pd.isna(ultimo_preco):
                    ultimo_preco = float(pd.to_numeric(df_prices_mk[ticker_unico], errors='coerce').dropna().iloc[-1])
                df_mark = pd.DataFrame(
                    {
                        'Pesos %': [100.0],
                        'Valor Alocado (R$)': [valor_total_markowitz],
                        'Ult. Preco': [ultimo_preco],
                        'Quantidade de Acoes': [int(np.floor(valor_total_markowitz / ultimo_preco)) if ultimo_preco > 0 else 0],
                    },
                    index=[ticker_unico],
                )
            else:
                volatility_mk = np.log(df_prices_mk.ffill()).diff().ewm(com=32).std()
                try:
                    _, _, df_mark_raw, _, _, _ = m.markowitz(df_prices_mk.copy(), volatility_mk, anos_cotacoes, 'RiskParity', 'CovarianceShrinkage', 'Sim', 'Sim', 'ema_historical_return', span=100, selic_aa=selic_aa_markowitz, cash=valor_total_markowitz, use_bcb=False)
                except Exception as exc:
                    st.warning(f'Nao foi possivel executar Markowitz para a cesta de acoes: {exc}')
                    return
                df_mark = df_mark_raw.iloc[:, :2].copy()
                df_mark.columns = ['Pesos %', 'Ult. Preco']
                # Garante o mesmo "Ult. Preco" mostrado na tabela de preços exibida na tela.
                df_mark['Ult. Preco'] = pd.to_numeric(df_mark.index.to_series().map(ultimos_precos_tela), errors='coerce').fillna(
                    pd.to_numeric(df_mark['Ult. Preco'], errors='coerce')
                )
                df_mark['Valor Alocado (R$)'] = valor_total_markowitz * (pd.to_numeric(df_mark['Pesos %'], errors='coerce').fillna(0.0) / 100)
                df_mark['Quantidade de Acoes'] = np.where(
                    pd.to_numeric(df_mark['Ult. Preco'], errors='coerce').fillna(0.0) > 0,
                    np.floor(df_mark['Valor Alocado (R$)'] / pd.to_numeric(df_mark['Ult. Preco'], errors='coerce').fillna(0.0)),
                    0,
                )
                df_mark['Quantidade de Acoes'] = pd.to_numeric(df_mark['Quantidade de Acoes'], errors='coerce').fillna(0).astype(int)
                df_mark = df_mark[['Pesos %', 'Valor Alocado (R$)', 'Ult. Preco', 'Quantidade de Acoes']]
            df_mark_alloc = df_mark.copy()
            total_row_mk = {'Pesos %': 0.0, 'Valor Alocado (R$)': 0.0, 'Ult. Preco': 0.0, 'Quantidade de Acoes': 0}
            for col in total_row_mk.keys():
                total_row_mk[col] = pd.to_numeric(df_mark[col], errors='coerce').fillna(0).sum()
            df_mark = pd.concat([df_mark, pd.DataFrame([total_row_mk], index=['TOTAL'])])
            st.dataframe(df_mark.round({'Pesos %': 2, 'Valor Alocado (R$)': 2, 'Ult. Preco': 2}), use_container_width=True)
            st.markdown('---')
            st.subheader('Oscilador para Acoes')
            df_prices_osc_acoes = df_prices_acoes.copy().ffill().dropna(how='all')
            if df_prices_osc_acoes.empty:
                st.warning('Sem dados validos para executar Oscilador nas acoes.')
                return
            df_pos_osc_acoes = m.oscilador(df_prices_osc_acoes).copy()
            if not df_pos_osc_acoes.empty:
                if 'pos_osc' not in df_pos_osc_acoes.columns:
                    df_pos_osc_acoes.columns = ['pos_osc']
                df_pos_osc_acoes = df_pos_osc_acoes.reindex(df_prices_osc_acoes.columns).rename_axis('Ticker').reset_index()
                df_pos_osc_acoes['pos_osc'] = pd.to_numeric(df_pos_osc_acoes['pos_osc'], errors='coerce').fillna(0.0)
            else:
                df_pos_osc_acoes = pd.DataFrame({'Ticker': list(df_prices_osc_acoes.columns), 'pos_osc': 0.0})
                st.warning('Oscilador nao retornou posicoes. Valores de pos_osc considerados como 0.')
            st.subheader('pos_osc por Ticker (Acoes)')
            st.dataframe(df_pos_osc_acoes[['Ticker', 'pos_osc']], use_container_width=True)
            st.markdown('---')
            st.subheader('Comparacao entre as Posicoes: Markowitz, Oscilator e Atual (Acoes)')
            if 'pos_atual' not in df_positions.columns:
                st.warning("Coluna obrigatoria nao encontrada em positions_BrunoBariotto: 'pos_atual'.")
                return
            df_pos_atual_acoes = df_positions[df_positions['Acao'].isin(ativos_acoes)][['Acao', 'pos_atual']].copy()
            df_pos_atual_acoes['pos_atual'] = pd.to_numeric(df_pos_atual_acoes['pos_atual'], errors='coerce').fillna(0.0)
            df_pos_atual_acoes.rename(columns={'Acao': 'Ticker', 'pos_atual': 'Posicao Atual'}, inplace=True)
            df_mark_cmp = df_mark_alloc.reset_index().rename(columns={'index': 'Ticker', 'Quantidade de Acoes': 'Posicao Markowitz', 'Ult. Preco': 'Ultimo Preco'})
            df_mark_cmp = df_mark_cmp[['Ticker', 'Posicao Markowitz', 'Ultimo Preco']]
            df_mark_cmp['Posicao Markowitz'] = pd.to_numeric(df_mark_cmp['Posicao Markowitz'], errors='coerce').fillna(0.0)
            df_mark_cmp['Ultimo Preco'] = pd.to_numeric(df_mark_cmp['Ultimo Preco'], errors='coerce').fillna(0.0)
            df_osc_cmp = df_pos_osc_acoes[['Ticker', 'pos_osc']].copy()
            df_osc_cmp['pos_osc'] = pd.to_numeric(df_osc_cmp['pos_osc'], errors='coerce').fillna(0.0)
            df_comp_acoes = pd.DataFrame({'Ticker': ativos_acoes})
            df_comp_acoes = df_comp_acoes.merge(df_pos_atual_acoes, on='Ticker', how='left')
            df_comp_acoes = df_comp_acoes.merge(df_mark_cmp, on='Ticker', how='left')
            df_comp_acoes = df_comp_acoes.merge(df_osc_cmp, on='Ticker', how='left')
            df_comp_acoes['Posicao Atual'] = pd.to_numeric(df_comp_acoes['Posicao Atual'], errors='coerce').fillna(0.0)
            df_comp_acoes['Posicao Markowitz'] = pd.to_numeric(df_comp_acoes['Posicao Markowitz'], errors='coerce').fillna(0.0)
            df_comp_acoes['Ultimo Preco'] = pd.to_numeric(df_comp_acoes['Ultimo Preco'], errors='coerce').fillna(0.0)
            df_comp_acoes['pos_osc'] = pd.to_numeric(df_comp_acoes['pos_osc'], errors='coerce').fillna(0.0)
            df_comp_acoes['Posicao Oscilator'] = df_comp_acoes['pos_osc'] * df_comp_acoes['Posicao Markowitz'] / 100
            df_comp_acoes['Valor Atual (R$)'] = df_comp_acoes['Posicao Atual'] * df_comp_acoes['Ultimo Preco']
            df_comp_acoes['Valor Markowitz (R$)'] = df_comp_acoes['Posicao Markowitz'] * df_comp_acoes['Ultimo Preco']
            df_comp_acoes['Valor Oscilator (R$)'] = df_comp_acoes['Posicao Oscilator'] * df_comp_acoes['Ultimo Preco']
            fig_comp_acoes = go.Figure(data=[go.Bar(name='Pos. Markowitz', x=df_comp_acoes['Ticker'], y=df_comp_acoes['Posicao Markowitz']), go.Bar(name='Pos. Oscilator', x=df_comp_acoes['Ticker'], y=df_comp_acoes['Posicao Oscilator']), go.Bar(name='Pos. Atual', x=df_comp_acoes['Ticker'], y=df_comp_acoes['Posicao Atual'])])
            fig_comp_acoes.update_layout(title={'text': 'Comparacao entre as Posicoes: Markowitz, Oscilator e Atual (Acoes)'}, xaxis_title='Acoes', yaxis_title='Quantidade', font=dict(family='Courier New, monospace', size=12), height=650)
            st.plotly_chart(fig_comp_acoes, use_container_width=True)
            st.markdown('---')
            st.subheader('Distribuicao por Acoes - Pizzas Comparativas (R$)')
            total_mark_acoes = float(df_comp_acoes['Valor Markowitz (R$)'].sum())
            total_osc_acoes = float(df_comp_acoes['Valor Oscilator (R$)'].sum())
            total_atual_acoes = float(df_comp_acoes['Valor Atual (R$)'].sum())
            for title, values in [(f'Posicao Markowitz - Soma Total: {total_mark_acoes:,.2f}', df_comp_acoes['Valor Markowitz (R$)']), (f'Posicao Oscilator - Soma Total: {total_osc_acoes:,.2f}', df_comp_acoes['Valor Oscilator (R$)']), (f'Posicao Atual - Soma Total: {total_atual_acoes:,.2f}', df_comp_acoes['Valor Atual (R$)'])]:
                fig_p = go.Figure(data=[go.Pie(labels=df_comp_acoes['Ticker'], values=values, textinfo='label+value+percent')])
                fig_p.update_layout(title={'text': title}, font=dict(family='Courier New, monospace', size=12), height=620, legend=dict(orientation='h', yanchor='top', y=-0.08, xanchor='center', x=0.5))
                st.plotly_chart(fig_p, use_container_width=True)
            st.markdown('---')
            st.subheader('Tabela Comparativa por Ticker (Acoes)')
            df_tab_acoes = df_comp_acoes[['Ticker', 'Posicao Atual', 'Posicao Markowitz', 'Posicao Oscilator']].copy()
            df_tab_acoes['Posicao Atual - Markowitz'] = df_tab_acoes['Posicao Atual'] - df_tab_acoes['Posicao Markowitz']
            df_tab_acoes['Posicao Final - Posicao Oscilator'] = df_tab_acoes['Posicao Atual'] - df_tab_acoes['Posicao Oscilator']
            total_row_acoes = {'Ticker': 'TOTAL'}
            for col in df_tab_acoes.columns:
                if col != 'Ticker':
                    total_row_acoes[col] = pd.to_numeric(df_tab_acoes[col], errors='coerce').fillna(0.0).sum()
            df_tab_acoes = pd.concat([df_tab_acoes, pd.DataFrame([total_row_acoes])], ignore_index=True)
            st.dataframe(df_tab_acoes, use_container_width=True)

            st.markdown('---')
            st.subheader('Tabela Comparativa por Ticker (Acoes) - Valores em R$')
            df_tab_acoes_rs = pd.DataFrame({
                'Ticker': df_comp_acoes['Ticker'],
                'Valor Atual (R$)': pd.to_numeric(df_comp_acoes['Valor Atual (R$)'], errors='coerce').fillna(0.0),
                'Valor Markowitz (R$)': pd.to_numeric(df_comp_acoes['Valor Markowitz (R$)'], errors='coerce').fillna(0.0),
                'Valor Oscilator (R$)': pd.to_numeric(df_comp_acoes['Valor Oscilator (R$)'], errors='coerce').fillna(0.0),
            })
            df_tab_acoes_rs['Valor Atual - Markowitz (R$)'] = df_tab_acoes_rs['Valor Atual (R$)'] - df_tab_acoes_rs['Valor Markowitz (R$)']
            df_tab_acoes_rs['Valor Final - Oscilator (R$)'] = df_tab_acoes_rs['Valor Atual (R$)'] - df_tab_acoes_rs['Valor Oscilator (R$)']

            total_row_acoes_rs = {'Ticker': 'TOTAL'}
            for col in df_tab_acoes_rs.columns:
                if col != 'Ticker':
                    total_row_acoes_rs[col] = pd.to_numeric(df_tab_acoes_rs[col], errors='coerce').fillna(0.0).sum()

            df_tab_acoes_rs = pd.concat([df_tab_acoes_rs, pd.DataFrame([total_row_acoes_rs])], ignore_index=True)
            st.dataframe(df_tab_acoes_rs.round(2), use_container_width=True)

    def posicao(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.title('Controle de Posição')
        st.markdown('---')
        #image = Image.open('alocation2.jpg')
        #st.image(image, caption='Alocação',width=400)
        #st.dataframe(df)

        df = self._normalize_positions_df(df)
        required_cols = {'Acao', 'pos_atual', 'pos_markw', 'pos_osc'}
        missing = required_cols.difference(set(df.columns))
        if missing:
            st.warning(f"Colunas obrigatórias ausentes: {', '.join(sorted(missing))}")
            st.write('Colunas encontradas:', list(df.columns))
            return
        
        acao_col = df.get('Acao')
        if acao_col is None:
            st.warning("Coluna obrigatória não encontrada: 'Acao'.")
            st.write('Colunas encontradas:', list(df.columns))
            return pd.DataFrame()

        m = Models()
        tickers = [str(t).strip() for t in acao_col if str(t).strip()]
        df_prices = m.download_prices_novo(tickers, per_data, anos_cotacoes, datas_inicio, datas_fim)
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
        usd_mask = df['Acao'] == 'USDBRL=X'
        if usd_mask.any():
            markowitz_usd = 30120 * df.loc[usd_mask, 'pos_osc'] / 100
        else:
            markowitz_usd = pd.Series([0.0])
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
        
        if ('HASH11.SA' in df['Acao'].values) and ('HASH11.SA' in df_prices.columns):
            df.loc[(df['Acao'] == 'HASH11.SA'), ['pos_markw']] = (
                df.loc[df['Acao'] == 'HASH11.SA', 'pos_markw'].iloc[0]
                + ((markowitz_btc) / (df_prices['HASH11.SA'].iloc[-1])).iloc[0]
            )
        else:
            st.warning('HASH11.SA não encontrado na planilha ou nos preços.')

        if ('IVVB11.SA' in df['Acao'].values) and ('IVVB11.SA' in df_prices.columns):
            df.loc[(df['Acao'] == 'IVVB11.SA'), ['pos_markw']] = (
                df.loc[df['Acao'] == 'IVVB11.SA', 'pos_markw'].iloc[0]
                + ((markowitz_spx) / (df_prices['IVVB11.SA'].iloc[-1])).iloc[0]
            )
        else:
            st.warning('IVVB11.SA não encontrado na planilha ou nos preços.')
        
        
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

        st.markdown('---')
        st.subheader('Alocação Necessária (PosxOsc x Último Preço)')

        df_alloc = df[['Acao', 'PosxOsc']].copy()
        df_alloc['Ultimo_Preco'] = df_alloc['Acao'].map(ultimo_preco)
        df_alloc['Valor_Alocacao'] = df_alloc['PosxOsc'] * df_alloc['Ultimo_Preco']

        if df_alloc['Ultimo_Preco'].isna().any():
            st.warning('Alguns tickers não possuem preço disponível no último dia.')

        st.dataframe(df_alloc.set_index('Acao').round({'PosxOsc': 2, 'Ultimo_Preco': 2, 'Valor_Alocacao': 2}))

        total_alocacao = df_alloc['Valor_Alocacao'].sum()
        st.metric('Total de Alocação Necessária (R$)', f"{total_alocacao:,.2f}")

        fig_alloc = go.Figure(
            data=[go.Bar(x=df_alloc['Acao'], y=df_alloc['Valor_Alocacao'])]
        )
        fig_alloc.update_layout(
            title={'text': 'Valor de Alocação Necessária por Ticker'},
            xaxis_title='Ticker',
            yaxis_title='R$',
            font=dict(family="Courier New, monospace", size=12),
            height=600,
        )
        st.plotly_chart(fig_alloc)
        
        
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
        df_prices = m.download_prices_novo(lista_acoes, per_data, anos_cotacoes, datas_inicio, datas_fim)
        
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
        
        df = df.copy()
        st.header('Carteira')
        st.markdown('---')
        
        df['ticker'] = df.ticker.apply(lambda l: l+".SA")
        
        m = Models()
        df_prices = m.download_prices_novo(list(df.ticker), per_data, anos_cotacoes, datas_inicio, datas_fim)
        st.write(df_prices)
        df_returns = m.returns(df_prices)
        
        df_info = pd.DataFrame(index=df.ticker)
        
        
        st.subheader('Métricas de Retorno')
        st.subheader('Retorno Anual Individual - Base 246 dias úteis [%]')
        df_info['Ret_Anual'] = df_returns.mean() * 246 * 100
        st.write(df_info['Ret_Anual'])
        
        #Last Price - to define alocation
        df_info['Last_price'] = df_prices.iloc[-1,:]
        
        
        st.write('% Alocação por ativo')

        df_info['%_aloc'] = df_prices.iloc[-1].values * df.qtdd.values /(df_prices.iloc[-1].values * df.qtdd.values).sum(axis=0)
        st.write(df_info['%_aloc'])
        
        st.write('Retorno Anual Carteira - Base 246 dias úteis [%]')
        pesos = np.array(list(df_info['%_aloc']))
        ret_esperado = np.dot(df_info['Ret_Anual'] , pesos)
        st.write(np.round(ret_esperado,4))
        
        st.write('Comparativo Carteira x IBOV')
        df_norm = pd.DataFrame(columns=df_prices.columns)
        df_norm = df_prices.apply(lambda x: x/x.iloc[0])
        st.write('DF Normalizado')
        df_norm['Carteira'] = df_norm.sum(axis=1).values/((df['qtdd'] != 0).sum())
        st.write(df_norm)
        
                
        fig_box = go.Figure()
        
        for col in df_norm.columns:
            fig_box.add_trace(go.Scatter(
                y=df_norm[col],
                x=df_norm.index,
                name=col))

        fig_box.update_layout(
            title={'text':'Preços Normalizados / Retorno'},
            width=900, height=600,
            xaxis_title='Date',
            yaxis_title='Preços Normalizados',
            font=dict(
                family="Courier New, monospace",
                size=18))

        fig_box.update_traces(marker={'size': 15})

        st.plotly_chart(fig_box)
        
        #df_norm.plot(figsize=(15,7), title='Histórico de Preços Normalizado')
        
        from scipy import stats
        
        st.subheader('Métricas de Risco')
        df_risk = pd.DataFrame(index=df.ticker)
        df_risk['var'] = df_returns.apply(lambda x: x.var())
        df_risk['std'] = df_returns.apply(lambda x: x.std()*100)
        df_risk['std_anual'] = df_returns.apply(lambda x: x.std()*np.sqrt(246)*100)
        df_risk['cv'] = df_returns.dropna().apply(lambda x: stats.variation(x))
        
        st.write(df_risk)
        
        st.write('Matriz de Correlação')
        st.write(df_returns.corr())

        var_carteira = (np.dot(pesos, np.dot(df_returns.cov() , pesos)))
        
        st.write('Desvio-Padrão da Carteira [%]')
        std_carteira = np.sqrt(var_carteira)*np.sqrt(246)*100
        st.write(np.round(std_carteira,4))
        
        st.subheader('Sharpe Ratio')
        st.write(f'Retorno Esperado {np.round(ret_esperado,4)}')
        st.write(f'Volatilidade {np.round(std_carteira,4)}')
        st.write('Sem Risco = 8.75%')
        st.write(f'Sharpe = {(ret_esperado - 8.75) / std_carteira}')
        
        st.header('CAPM')
        st.write('Comparação dos Riscos e Retornos obtidos por um ativo/carteira comparado a um Benchmark (IBOV)')
        
        st.write(r'''
                 $$ 
                 R_i = R_f + \beta * (R_m - R_f) 
                 $$
                 ''')
        st.write('Em que:')
        st.write(r'''
                 $$
                 R_i 
                 $$ 
                 : Retorno CAPM esperado para o ativo/carteira para correr o risco de investir''')
        st.write(r'''
                 $$
                 R_f 
                 $$ 
                 : Retorno Benchmark i.e IBOVESPA''')
        st.write(r'''
                 $$
                 \beta 
                 $$ 
                 : Comparação do risco, = 1: Forte correlação com IBOV, > 1 : maior de volatilidade, < 1 : menor volatilidade''')
        st.write(r'''
                 $$ 
                 R_m 
                 $$
                 : Retorno Risk Free, i.e Tesouro Direto/CDB''')
                 
        df_capm = df_returns.copy().dropna()

        df_capm = df_capm.apply(lambda x: np.polyfit(df_capm.iloc[:,-1], x , deg=1))
        df_capm.index = ['Beta', 'Alfa']
        df_capm_f = pd.DataFrame()
        df_capm_f = df_capm.T
        df_capm_f['Alfa'] = df_capm_f['Alfa'] * 100
        
        df_capm_f['CAPM_Ret%'] = (0.085 + df_capm_f['Beta'] * (df_returns.copy().dropna().mean().T - 0.085))*100
        st.write('Vol Benchmark IBOV')
        st.write(np.round(df_returns.iloc[:,-1].dropna().std()*np.sqrt(246),4))
        df_capm_f['CAPM_Sharpe'] = df_capm_f['CAPM_Ret%'] / (df_returns.iloc[:,-1].dropna().std()*np.sqrt(246) * 100 * df_capm_f['Beta'])
        st.write(df_capm_f)
    
    # modelos
    # Tela que exibe dados dos outputs para os modelos de Markwitz e Oscilador
    #
    def modelos(self, df, per_data, anos_cotacoes, datas_inicio, datas_fim):
        st.header('Modelos')
        st.markdown('---')

        df = self._normalize_positions_df(df)
        if 'Acao' not in df.columns:
            st.warning("Coluna obrigatória não encontrada: 'Acao'.")
            st.write('Colunas encontradas:', list(df.columns))
            return pd.DataFrame()
        
        m = Models()
        df_prices = m.download_prices_novo(list(df.Acao), per_data, anos_cotacoes, datas_inicio, datas_fim)
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
        
        tipo = st.radio('Escolha o tipo: ', ['Ações', 'Fundos Imobiliários'], key=131554)
        
        if tipo == 'Ações':
            st.title('Fundamento de Ações')
            
            st.header('Indicadores mais recentes')
            
            st.write('Atualizar: acessar https://statusinvest.com.br/acoes/busca-avancada , clicar em buscar e fazer download do csv')
            import os
            #import openpyxl
            st.write(os.getcwd())
            st.write('Salvar a planilha em C:/Users/Dell inspiron/app_positions')
            
            ind_df = pd.read_csv('statusinvest-busca-avancada.csv', sep=';')
            
            setores_df = pd.read_csv('setores.csv',  encoding='latin1')
            st.write(setores_df)
            
            ind_df_final = setores_df.merge(ind_df, left_on='TICKER',right_on='TICKER')
            
            st.write(ind_df_final)
            
            for c in list(ind_df_final.columns[5:]):
                ind_df_final[c] = ind_df_final[c].fillna(0).apply(lambda x: pd.to_numeric(str(x).replace('.','').replace(',','.')))
                
            st.write(ind_df_final)
            
            radio = st.radio('Analise por Setor ou Empresas semelhantes?', ['Setor','Empresas'])
            
            if radio == 'Setor':
                setor = st.selectbox('Escolha o Segmento:', ind_df_final['SEGMENTO'].unique())
                
                st.write(ind_df_final[(ind_df_final['SEGMENTO']==setor)]['TICKER'].unique())
                
                todas = ind_df_final[(ind_df_final['SEGMENTO']==setor)]['TICKER'].unique()
                
                empresas = st.multiselect('Selecione as empresas', todas)
                
                st.write(ind_df_final[ind_df_final['TICKER'].isin(empresas)].iloc[:,4:])
                
                st.header('Curso')
                st.write(ind_df_final.columns)
                st.write('Filtro 1) Margem EBIT > 0 para garantir que Div Liq/Ebit menor melhor')
                
                st.subheader('Indicadores de Valuation: P/EBIT e DY')
                val_df = ind_df_final[ind_df_final['SEGMENTO'] == setor][['TICKER','P/EBIT','DY','P/L','P/VP','P/ATIVOS']]
                st.write(val_df[(val_df['P/EBIT'] >= 0) & (val_df['P/L'] >= 0) & (val_df['P/VP'] >= 0) & (val_df['P/ATIVOS'] >= 0) ])
                
                st.subheader('Indicadores de Rentabilidade: Margem EBIT e ROIC')
                st.write(ind_df_final[ind_df_final['SEGMENTO'] == setor][['TICKER','MARGEM EBIT','ROIC','MARGEM BRUTA','MARG. LIQUIDA','ROE','ROA']])
                
                st.subheader('Indicadores de Endividamento: DIVIDA LIQUIDA / EBIT, LIQ. CORRENTE')
                st.write(ind_df_final[ind_df_final['SEGMENTO'] == setor][['TICKER','DIVIDA LIQUIDA / EBIT','LIQ. CORRENTE',' LIQUIDEZ MEDIA DIARIA']])

                
                
            if radio == 'Empresas':
                empresa = st.selectbox('Escolha a Empresa:', ind_df_final['TICKER'].unique())
                
                segmento = ind_df_final[(ind_df_final['TICKER']==empresa)]['SEGMENTO'].iloc[0]
                st.write(segmento)
                
                st.write(ind_df_final[ind_df_final['SEGMENTO'] == segmento].iloc[:,4:])
                
                st.header('Curso')
                st.write(ind_df_final.columns)
                st.write('Filtro 1) Margem EBIT > 0 para garantir que Div Liq/Ebit menor melhor')
                
                st.subheader('Indicadores de Valuation: P/EBIT e DY')
                val_df = ind_df_final[ind_df_final['SEGMENTO'] == segmento][['TICKER','P/EBIT','DY','P/L','P/VP','P/ATIVOS']]
                filter_df = val_df[(val_df['P/EBIT'] >= 0) & (val_df['P/L'] >= 0) & (val_df['P/VP'] >= 0) & (val_df['P/ATIVOS'] >= 0) ]
                st.write(filter_df)
                filter_df['P/EBIT_rank'] = filter_df['P/EBIT'].rank()
                filter_df['DY_rank'] =  filter_df['DY'].rank(ascending=False)
                filter_df['P/L_rank'] =  filter_df['P/L'].rank()
                filter_df['P/VP_rank'] =   filter_df['P/VP'].rank()
                filter_df['P/ATIVOS_rank'] =   filter_df['P/ATIVOS'].rank()
                filter_df['rank_sum'] = filter_df['P/EBIT_rank'] + filter_df['P/L_rank'] + filter_df['P/VP_rank'] + filter_df['P/ATIVOS_rank'] + filter_df['DY_rank'] 
                st.write(filter_df[['TICKER','P/EBIT_rank','DY_rank','P/L_rank','P/VP_rank','P/ATIVOS_rank','rank_sum']].sort_values('rank_sum', ascending = True))
                #st.write(filter_df[['P/EBIT','P/L','P/VP','P/ATIVOS']].apply(tuple, axis=1).rank(method='dense', ascending=False).astype(int))


                
                
                st.subheader('Indicadores de Rentabilidade: Margem EBIT e ROIC')
                st.write(ind_df_final[ind_df_final['SEGMENTO'] == segmento][['TICKER','MARGEM EBIT','ROIC','MARGEM BRUTA','MARG. LIQUIDA','ROE','ROA']])
                
                st.subheader('Indicadores de Endividamento: DIVIDA LIQUIDA / EBIT, LIQ. CORRENTE')
                st.write(ind_df_final[ind_df_final['SEGMENTO'] == segmento][['TICKER','DIVIDA LIQUIDA / EBIT','LIQ. CORRENTE',' LIQUIDEZ MEDIA DIARIA']])

                    
            
            
        
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
                
            

        
            
        
            

        
    
    
        
        

