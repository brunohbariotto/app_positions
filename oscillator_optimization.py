"""Per-asset, causal oscillator research. No UI, network or legacy mutations."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

VERSION = 'osc-asset-v1'
SHEET = 'Oscillator_Windows'
WEIGHTS = {
    'slow': (10., 20., 35., 50.),
    'medium': (7.5, 15., 30., 40.),
    'fast': (5., 10., 25., 35.),
}
MIN_QUANTILES = 64


@dataclass(frozen=True)
class Windows:
    # Each value is the longer EMA time constant; the shorter one is days / 3.
    slow: int = 96
    medium: int = 48
    fast: int = 24

    def __post_init__(self):
        values = (self.slow, self.medium, self.fast)
        if any(not np.isfinite(v) or int(v) != v or v % 3 or not 6 <= v <= 756 for v in values):
            raise ValueError('Janelas devem ser inteiras, múltiplas de 3, entre 6 e 756.')
        if not self.fast < self.medium < self.slow:
            raise ValueError('É necessário Fast < Medium < Slow.')
        for kind in WEIGHTS:
            object.__setattr__(self, kind, int(getattr(self, kind)))


DEFAULT_WINDOWS = Windows()


def clean_prices(prices):
    prices = pd.to_numeric(prices, errors='coerce').replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep='last')]
    if len(prices) < 3 or (prices <= 0).any():
        raise ValueError('Histórico insuficiente ou preços não positivos.')
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError('O histórico deve ter datas no índice.')
    if prices.nunique() < 3:
        raise ValueError('Histórico sem variação suficiente.')
    return prices.astype(float)


def signal_positions(prices, days):
    """Same EWM/tanh/volatility formula as PL_osc, with a one-asset norm."""
    volatility = np.log(prices).diff().ewm(com=32).std()
    scaled_prices = (np.log(prices).diff() / volatility).clip(-5, 5).cumsum()
    fast, slow = days // 3, days
    f, g = 1 - 1 / fast, 1 - 1 / slow
    norm = np.sqrt(1 / (1 - f*f) - 2 / (1 - f*g) + 1 / (1 - g*g))
    osc = (scaled_prices.ewm(span=2*fast-1).mean()
           - scaled_prices.ewm(span=2*slow-1).mean()) / norm
    position = (np.tanh(osc) / volatility).clip(-50, 50).fillna(0.)
    return (position / position.abs() / volatility).clip(-50, 50).fillna(0.)


def component_weights(position, kind, inverted=False):
    # At t, thresholds use observations strictly before t.
    past = position.expanding(min_periods=MIN_QUANTILES)
    q1, q2, q3 = (past.quantile(q).shift(1) for q in (.25, .5, .75))
    weights = WEIGHTS[kind][::-1] if inverted else WEIGHTS[kind]
    result = pd.Series(np.select(
        [position >= q3, position >= q2, position >= q1],
        weights[:3], default=weights[3]), index=position.index)
    return result.where(q1.notna())


def allocation_history(prices, windows=DEFAULT_WINDOWS, inverted=False):
    prices = clean_prices(prices)
    parts = [component_weights(signal_positions(prices, getattr(windows, kind)), kind, inverted)
             for kind in WEIGHTS]
    return sum(parts).rename('pos_osc')


def strategy_returns(prices, allocation, cost_bps=10.):
    if not np.isfinite(cost_bps) or not 0 <= cost_bps <= 1000:
        raise ValueError('Custo deve estar entre 0 e 1000 pontos-base.')
    # Signal at close t, execute at close t+1, first earned return ends at t+2.
    held = allocation.shift(2).fillna(0.) / 100
    turnover = held.diff().abs().fillna(held.abs())
    return held * prices.pct_change(fill_method=None).fillna(0.) - turnover * cost_bps / 10000


def metrics(returns):
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    if equity.empty:
        raise ValueError('Período de avaliação vazio.')
    drawdown = equity / equity.cummax().clip(lower=1.) - 1
    return {'retorno': float(equity.iloc[-1] - 1), 'drawdown': float(drawdown.min())}


def candidate_grid(slow, medium, fast):
    candidates = {DEFAULT_WINDOWS}
    for s, m, f in product(slow, medium, fast):
        # Invalid ordering is filtered, malformed bounds still raise.
        if f < m < s:
            candidates.add(Windows(s, m, f))
    return sorted(candidates, key=lambda w: (w.slow, w.medium, w.fast))


def timing_metrics(prices, allocation, horizon=5):
    # Changes decided at t execute at t+1; measure t+1 -> t+1+horizon.
    future = prices.shift(-(horizon + 1)) / prices.shift(-1) - 1
    change = allocation.diff()
    mask = change.ne(0) & change.notna() & future.notna()
    correct = (change[mask] * future[mask]) > 0
    return {'decisoes': int(mask.sum()),
            'acerto': float(correct.mean()) if mask.any() else np.nan}


def optimize_asset(prices, candidates, cost_bps=10., test_fraction=.2,
                   validation_fraction=.2, shortlist=10, horizon=5):
    prices = clean_prices(prices)
    candidates = sorted(set(candidates) | {DEFAULT_WINDOWS},
                        key=lambda w: (w.slow, w.medium, w.fast))
    if not 0 < test_fraction < .5 or not 0 < validation_fraction < .5:
        raise ValueError('Frações de validação e teste inválidas.')
    if test_fraction + validation_fraction >= .8 or shortlist < 1 or horizon < 1:
        raise ValueError('Configuração de avaliação inválida.')
    # Common warm-up ensures every candidate sees exactly the same evaluation dates.
    warmup = max(3 * max(w.slow for w in candidates), MIN_QUANTILES + 2)
    usable = len(prices) - warmup
    if usable < 315:
        raise ValueError(f'Histórico curto: {len(prices)} observações; são necessárias '
                         f'{warmup + 315} para esta grade (inclui aquecimento).')
    train_end = warmup + int(usable * (1 - test_fraction - validation_fraction))
    validation_end = len(prices) - int(usable * test_fraction)
    if min(train_end - warmup, validation_end - train_end, len(prices) - validation_end) < 63:
        raise ValueError('Cada período deve ter pelo menos 63 observações.')
    parts = {}
    for kind in WEIGHTS:
        for days in sorted({getattr(w, kind) for w in candidates}):
            parts[kind, days] = component_weights(signal_positions(prices, days), kind)
    def allocation(w):
        return sum(parts[kind, getattr(w, kind)] for kind in WEIGHTS).rename('pos_osc')
    def returns(w):
        return strategy_returns(prices, allocation(w), cost_bps)
    # Selection never reads test returns. Deterministic tie break favors current windows.
    ranking = []
    for w in candidates:
        result = metrics(returns(w).iloc[warmup:train_end])
        ranking.append((w, result['retorno']))
    ranking.sort(key=lambda item: (-item[1], item[0] != DEFAULT_WINDOWS,
                                  item[0].slow, item[0].medium, item[0].fast))
    finalists = list(dict.fromkeys([w for w, _ in ranking[:shortlist]] + [DEFAULT_WINDOWS]))
    winner = max(finalists, key=lambda w: (
        metrics(returns(w).iloc[train_end:validation_end])['retorno'], w == DEFAULT_WINDOWS))
    selected, baseline = returns(winner), returns(DEFAULT_WINDOWS)
    buy_hold = prices.pct_change(fill_method=None).fillna(0.)
    records = []
    for name, start, end in [('Treino', warmup, train_end),
                             ('Validação', train_end, validation_end),
                             ('Teste', validation_end, len(prices))]:
        for label, series in [('Otimizado', selected), ('Padrão causal', baseline),
                              ('Comprar e manter', buy_hold)]:
            segment = series.iloc[start:end].copy()
            # Oscillator segments continue the simulated holdings across split boundaries.
            if label == 'Comprar e manter':
                segment.iloc[0] -= cost_bps / 10000
            records.append({'periodo': name, 'estrategia': label,
                            'inicio': prices.index[start].date().isoformat(),
                            'fim': prices.index[end - 1].date().isoformat(), **metrics(segment)})
    evaluation = pd.DataFrame(records)
    test = evaluation[evaluation.periodo.eq('Teste')].set_index('estrategia')
    path = pd.DataFrame({'Otimizado': selected, 'Padrão causal': baseline,
                         'Comprar e manter': buy_hold}).iloc[validation_end:].copy()
    path.iloc[0, path.columns.get_loc('Comprar e manter')] -= cost_bps / 10000
    timing = timing_metrics(prices.iloc[validation_end - 1:],
                            allocation(winner).iloc[validation_end - 1:], horizon)
    row = {
        'slow_dias': winner.slow, 'medium_dias': winner.medium, 'fast_dias': winner.fast,
        'versao': VERSION, 'observacoes': len(prices), 'combinacoes': len(candidates),
        'inicio_historico': prices.index[0].date().isoformat(),
        'fim_historico': prices.index[-1].date().isoformat(),
        'inicio_teste': prices.index[validation_end].date().isoformat(),
        'custo_bps': cost_bps, 'horizonte': horizon,
        'retorno_teste': test.loc['Otimizado', 'retorno'],
        'retorno_padrao_teste': test.loc['Padrão causal', 'retorno'],
        'drawdown_teste': test.loc['Otimizado', 'drawdown'],
        'excesso_teste': test.loc['Otimizado', 'retorno'] - test.loc['Padrão causal', 'retorno'],
        'acerto_teste': timing['acerto'], 'decisoes_teste': timing['decisoes'],
        'otimizado_em': pd.Timestamp.now(tz='UTC').isoformat(),
        'grade_slow': ','.join(map(str, sorted({w.slow for w in candidates}))),
        'grade_medium': ','.join(map(str, sorted({w.medium for w in candidates}))),
        'grade_fast': ','.join(map(str, sorted({w.fast for w in candidates}))),
        'fracao_teste': test_fraction, 'fracao_validacao': validation_fraction,
        'finalistas': shortlist,
        'objetivo': 'retorno_liquido_validacao', 'aquecimento': warmup,
    }
    return {'row': row, 'evaluation': evaluation, 'equity': (1 + path).cumprod(),
            'allocation': allocation(winner).iloc[validation_end:]}


def saved_windows(frame):
    valid, errors = {}, []
    if frame.empty:
        return valid, errors
    required = {'Ticker', 'slow_dias', 'medium_dias', 'fast_dias', 'versao'}
    if not required.issubset(frame.columns):
        return {}, ['A aba de janelas não possui todas as colunas obrigatórias.']
    for _, row in frame.iterrows():
        ticker = str(row['Ticker']).strip()
        try:
            if row['versao'] != VERSION or ticker == 'Taxa_Juros_Brasil' or not ticker:
                raise ValueError('Versão ou ativo incompatível.')
            valid[ticker] = Windows(*(float(row[f'{kind}_dias']) for kind in WEIGHTS))
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(f'{ticker}: {exc}')
    return valid, errors


def merge_saved(existing, updates):
    """Upsert by ticker; keep unrelated assets and unknown spreadsheet columns."""
    if updates.empty:
        return existing.copy()
    if 'Ticker' not in updates or updates['Ticker'].duplicated().any():
        raise ValueError('Resultados devem ter um único registro por ticker.')
    if existing.empty:
        return updates.copy()
    if 'Ticker' not in existing or existing['Ticker'].duplicated().any():
        raise ValueError('A aba existente não possui tickers únicos; corrija antes de salvar.')
    old = existing.set_index('Ticker')
    new = updates.set_index('Ticker')
    combined = old.reindex(index=old.index.union(new.index, sort=False),
                          columns=old.columns.union(new.columns, sort=False))
    combined.loc[new.index, new.columns] = new
    return combined.reset_index()
