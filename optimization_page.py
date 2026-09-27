"""Streamlit interface and maximum-history provider for oscillator research."""
import pandas as pd
import streamlit as st
import yfinance as yf

from oscillator_optimization import (
    SHEET, Windows, allocation_history, candidate_grid, clean_prices,
    optimize_asset, saved_windows,
)


@st.cache_data(ttl=21600, show_spinner=False)
def maximum_history(ticker):
    symbol = {'^BVSP.SA': '^BVSP'}.get(ticker, ticker)
    if symbol == 'Taxa_Juros_Brasil':
        raise ValueError('Taxa de juros não é uma série de preços negociáveis.')
    # Download independently: never truncate an older asset to a newer asset's IPO.
    history = yf.Ticker(symbol).history(period='max', interval='1d', auto_adjust=False,
                                        actions=False, raise_errors=True, timeout=30)
    if history.empty or 'Adj Close' not in history:
        raise ValueError('O provedor não retornou preços ajustados.')
    prices = history['Adj Close'].copy()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.normalize()
    today = pd.Timestamp.now(tz='America/Sao_Paulo').tz_localize(None).normalize()
    prices = prices.loc[prices.index < today]
    return clean_prices(prices).rename(ticker)


def allocation_with_saved_windows(model, prices, configuration, as_of=None):
    """Opt-in only; failed/unconfigured assets retain the original basket result."""
    result = model.oscilador(prices).copy()
    windows, errors = saved_windows(configuration)
    for error in errors:
        st.warning(error)
    applied = []
    for ticker in prices.columns:
        if ticker not in windows:
            continue
        try:
            saved_row = configuration.loc[configuration['Ticker'].astype(str).str.strip().eq(ticker)].iloc[-1]
            if as_of is not None and pd.notna(saved_row.get('fim_historico')):
                if pd.Timestamp(saved_row['fim_historico']) > pd.Timestamp(as_of):
                    raise ValueError('As janelas foram pesquisadas com dados posteriores à data final selecionada.')
            history = maximum_history(ticker)
            if as_of is not None:
                history = history.loc[history.index <= pd.Timestamp(as_of)]
            w = windows[ticker]
            if len(history) < max(3 * w.slow, 66):
                raise ValueError('Histórico insuficiente para aquecer as janelas salvas.')
            value = allocation_history(history, w).iloc[-1]
            if pd.isna(value):
                raise ValueError('Não foi possível calcular o percentual atual.')
            result.loc[ticker, 'pos_osc'] = value
            applied.append({'Ticker': ticker, 'slow_dias': w.slow, 'medium_dias': w.medium,
                            'fast_dias': w.fast, 'pos_osc': value,
                            'ultima_cotacao': history.index[-1].date()})
        except Exception as exc:
            st.warning(f'{ticker}: mantida a regra atual. {exc}')
    if applied:
        st.caption('Janelas otimizadas aplicadas abaixo. Os gráficos anteriores mostram a regra atual. '
                   'O cálculo otimizado usa todo o histórico disponível até a data final, '
                   'quartis causais e normalização individual por ativo.')
        st.dataframe(pd.DataFrame(applied), use_container_width=True)
    unchanged = [ticker for ticker in prices.columns if ticker not in {r['Ticker'] for r in applied}]
    if unchanged:
        st.info('Regra atual mantida para: ' + ', '.join(unchanged))
    return result


def render_optimization_page(google, tickers):
    st.title('Otimização de Janelas')
    st.write('Busca por ativo dos períodos Slow, Medium e Fast, mantendo os percentuais '
             'de alocação de cada quartil. O melhor resultado é limitado à grade pesquisada.')
    st.caption('Esta página usa o histórico máximo disponível no Yahoo, independentemente '
               'da janela de datas do menu lateral. Dias representam observações de negociação '
               '(dias corridos para criptomoedas), sem preencher feriados ou fins de semana.')
    with st.expander('Como a otimização é avaliada'):
        st.write('As janelas informadas são o período maior de cada par de médias: '
                 'Slow 32/96, Medium 16/48 e Fast 8/24 no padrão atual. A proporção 1:3 é mantida. '
                 'A busca preserva Fast < Medium < Slow e os pesos originais, cuja soma pode chegar a 125%. '
                 'Cada ativo é normalizado individualmente, sem depender da composição da carteira.')
        st.write('Os quartis usam somente observações anteriores ao sinal. O sinal do fechamento '
                 'é executado no fechamento seguinte e passa a receber retornos depois disso. '
                 'Os 60% iniciais selecionam até dez candidatos pelo retorno líquido; '
                 'os 20% seguintes escolhem o vencedor. Os 20% finais são um teste separado, '
                 'que não escolhe as janelas. Há um aquecimento comum de três vezes a maior janela Slow.')
        st.write('O comparativo “Padrão causal” usa 96/48/24 com a mesma normalização individual '
                 'e os mesmos quartis causais; não reproduz o gráfico histórico legado, que usa '
                 'quartis de toda a amostra e normalização da cesta. A simulação rebalanceia a '
                 'exposição percentual, com custos sobre mudanças de exposição, caixa sem remuneração '
                 'e sem impostos ou custo de financiamento. Não simula a evolução do Markowitz.')
        st.markdown('Referências: [avaliação temporal](https://sklearn.org/stable/modules/cross_validation.html#time-series-split) '
                    'e [histórico de preços](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html).')
    tickers = sorted({str(t).strip() for t in tickers if str(t).strip() and str(t).lower() != 'nan'})
    if 'Taxa_Juros_Brasil' in tickers:
        st.info('Taxa_Juros_Brasil mantém a regra atual: otimizar a variação de uma taxa como '
                'se fosse retorno de um ativo produziria uma comparação incorreta. '
                'Sua otimização exige definir uma série de retorno investível correspondente.')
        tickers.remove('Taxa_Juros_Brasil')
    with st.form('oscillator_search'):
        selected = st.multiselect('Ativos', tickers, default=tickers[:1])
        columns = st.columns(3)
        slow = columns[0].text_input('Slow (dias, separados por vírgula)', '60,96,144,192,252')
        medium = columns[1].text_input('Medium (dias)', '24,36,48,72,96')
        fast = columns[2].text_input('Fast (dias)', '9,12,18,24,36')
        cost = st.number_input('Custo por mudança de exposição (pontos-base; 10 = 0,10%)',
                               min_value=0., max_value=1000., value=10., step=1.)
        horizon = st.number_input('Observações seguintes para medir acerto de aumentos/reduções',
                                  min_value=1, max_value=63, value=5)
        run = st.form_submit_button('Executar otimização')
    if run:
        st.session_state.pop('oscillator_results', None)
        try:
            grids = [[int(value.strip()) for value in text.split(',')] for text in (slow, medium, fast)]
            if any(not 6 <= n <= 756 or n % 3 for grid in grids for n in grid):
                raise ValueError('Use múltiplos de 3 entre 6 e 756.')
            if any(len(grid) > 15 for grid in grids):
                raise ValueError('Use no máximo 15 valores por janela.')
            if not any(f < m < s for s in grids[0] for m in grids[1] for f in grids[2]):
                raise ValueError('A grade precisa conter pelo menos uma combinação Fast < Medium < Slow.')
            candidates = candidate_grid(*grids)
            if len(candidates) > 1500:
                raise ValueError('Reduza a grade para até 1.500 combinações.')
            if not selected:
                raise ValueError('Selecione pelo menos um ativo.')
        except ValueError as exc:
            st.error(str(exc))
        else:
            results = {}
            progress = st.progress(0., text='Buscando históricos e avaliando janelas...')
            for i, ticker in enumerate(selected):
                try:
                    with st.spinner(f'{ticker}: histórico máximo e {len(candidates)} combinações...'):
                        data = maximum_history(ticker)
                        results[ticker] = optimize_asset(data, candidates, cost_bps=cost, horizon=int(horizon))
                        results[ticker]['row']['Ticker'] = ticker
                except Exception as exc:
                    st.warning(f'{ticker}: {exc}')
                progress.progress((i + 1) / len(selected), text=f'{ticker} concluído')
            st.session_state['oscillator_results'] = results
    results = st.session_state.get('oscillator_results', {})
    if results:
        rows = pd.DataFrame([value['row'] for value in results.values()])
        st.subheader('Janelas encontradas por ativo')
        display = ['Ticker', 'slow_dias', 'medium_dias', 'fast_dias', 'observacoes',
                   'inicio_historico', 'fim_historico', 'retorno_teste', 'retorno_padrao_teste',
                   'excesso_teste', 'drawdown_teste', 'acerto_teste', 'decisoes_teste']
        st.dataframe(rows[display], use_container_width=True)
        st.caption('Retornos, drawdown e acerto em fração: 0,10 = 10%. '
                   'Acerto mede se aumentar/reduzir a exposição antecedeu alta/baixa após a execução.')
        if rows['excesso_teste'].le(0).any():
            st.warning('Há ativos sem melhora sobre o padrão causal no teste. '
                       'As janelas continuam visíveis para análise; resultados passados não garantem melhora futura.')
        ticker = st.selectbox('Detalhar ativo', list(results))
        detail = results[ticker]
        st.dataframe(detail['evaluation'], use_container_width=True)
        st.caption('Evolução de uma unidade de capital no teste separado')
        st.line_chart(detail['equity'])
        st.caption('Alocação decidida no fechamento (%) durante o teste')
        st.line_chart(detail['allocation'])
        st.download_button('Baixar resultados CSV', rows.to_csv(index=False).encode('utf-8-sig'),
                           file_name='janelas_otimizadas.csv', mime='text/csv')
        to_save = st.multiselect('Ativos cujas janelas serão salvas', list(results), default=list(results))
        if st.button('Salvar janelas no Position_Control', disabled=not to_save):
            try:
                google.save_oscillator_windows(rows[rows.Ticker.isin(to_save)])
                st.success(f'Janelas salvas na aba {SHEET}. Ative a opção em Alocação por Ativo para usá-las.')
            except Exception as exc:
                st.error(f'Não foi possível salvar as janelas: {exc}')
    st.subheader('Janelas salvas no Position_Control')
    try:
        saved = google.read_oscillator_windows()
        if saved.empty:
            st.info('Nenhuma janela salva ainda.')
        else:
            st.dataframe(saved, use_container_width=True)
    except Exception as exc:
        st.warning(f'Não foi possível consultar as janelas salvas: {exc}')
