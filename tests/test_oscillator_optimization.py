import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oscillator_optimization import (
    DEFAULT_WINDOWS, VERSION, Windows, allocation_history, candidate_grid,
    clean_prices, component_weights, merge_saved, metrics, optimize_asset,
    saved_windows, signal_positions, strategy_returns, timing_metrics,
)


def prices(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(.0003, .017, n))),
                     index=pd.bdate_range('2010-01-01', periods=n), name='TEST')


def test_signal_matches_legacy_single_asset_formula():
    # Execute the actual legacy methods without importing its UI/network dependencies.
    tree = ast.parse(Path('models.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Models')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'osc', 'PL_osc'}]
    namespace = {'np': np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<legacy>', 'exec'), namespace)
    p = prices().to_frame()
    vol = np.log(p).diff().ewm(com=32).std()
    transformed = (np.log(p).diff() / vol).clip(-5, 5).cumsum()
    for days in [24, 48, 96, 144]:
        expected, _, _ = namespace['Models']().PL_osc(p, transformed, vol, days // 3, days)
        pd.testing.assert_series_equal(signal_positions(p.TEST, days), expected.TEST)


def test_causal_signals_do_not_change_when_future_is_added():
    p = prices()
    short = allocation_history(p.iloc[:900])
    full = allocation_history(p)
    pd.testing.assert_series_equal(short, full.iloc[:900])


@pytest.mark.parametrize('kind,expected', [
    ('slow', [50, 35, 20, 10]), ('medium', [40, 30, 15, 7.5]), ('fast', [35, 25, 10, 5])])
def test_quartile_weights_preserved(kind, expected):
    for value, weight in zip([0, 24, 40, 64], expected):
        p = pd.Series(list(range(64)) + [value], dtype=float)
        assert component_weights(p, kind).iloc[-1] == weight
        assert component_weights(p, kind).iloc[:64].isna().all()


def test_execution_lag_and_turnover_cost():
    p = pd.Series([100., 100., 100., 110., 121., 121.])
    allocation = pd.Series([0., 100., 50., 50., 50., 50.])
    r = strategy_returns(p, allocation, 10.)
    np.testing.assert_allclose(r, [0., 0., 0., .099, .0495, 0.], atol=1e-12)


def test_future_test_prices_cannot_select_windows():
    p = prices()
    grid = candidate_grid([60, 96, 144], [36, 48], [12, 24])
    original = optimize_asset(p, grid)
    altered = p.copy()
    test_start = pd.Timestamp(original['row']['inicio_teste'])
    count = (p.index >= test_start).sum()
    altered.loc[test_start:] *= np.exp(np.linspace(0, 2, count))
    rerun = optimize_asset(altered, grid)
    for col in ['slow_dias', 'medium_dias', 'fast_dias']:
        assert original['row'][col] == rerun['row'][col]
    assert original['row']['retorno_teste'] != rerun['row']['retorno_teste']
    assert len(original['evaluation']) == 9


def test_short_history_and_non_prices_are_rejected():
    with pytest.raises(ValueError, match='Histórico curto'):
        optimize_asset(prices(100), [DEFAULT_WINDOWS])
    with pytest.raises(ValueError, match='variação'):
        clean_prices(pd.Series(100., index=pd.bdate_range('2020-01-01', periods=400)))


def test_drawdown_includes_initial_loss():
    assert metrics(pd.Series([-.1, 0.]))['drawdown'] == pytest.approx(-.1)


def test_timing_measured_after_execution():
    p = pd.Series([100., 1., 10., 20., 30., 40.])
    allocation = pd.Series([10., 20., 20., 20., 20., 20.])
    result = timing_metrics(p, allocation, horizon=1)
    assert result == {'decisoes': 1, 'acerto': 1.}


def test_configuration_roundtrip_and_upsert_preserve_other_assets():
    existing = pd.DataFrame([{'Ticker': 'A', 'slow_dias': 96, 'medium_dias': 48,
                             'fast_dias': 24, 'versao': VERSION, 'notes': 'keep'},
                            {'Ticker': 'B', 'slow_dias': 96, 'medium_dias': 48,
                             'fast_dias': 24, 'versao': VERSION, 'notes': 'other'}])
    updated = existing.iloc[:1].drop(columns='notes').copy()
    updated['slow_dias'] = 144
    merged = merge_saved(existing, updated)
    config, errors = saved_windows(merged)
    assert not errors
    assert config['A'] == Windows(144, 48, 24)
    assert config['B'] == DEFAULT_WINDOWS
    assert merged.set_index('Ticker').loc['A', 'notes'] == 'keep'
    assert merged.set_index('Ticker').loc['B', 'notes'] == 'other'


def test_invalid_saved_windows_are_not_applied():
    frame = pd.DataFrame([{'Ticker': 'A', 'slow_dias': 30.5, 'medium_dias': 48,
                           'fast_dias': 24, 'versao': VERSION}])
    config, errors = saved_windows(frame)
    assert not config and errors


def test_grid_is_ordered_and_contains_baseline():
    grid = candidate_grid([60, 96], [24, 48, 96], [12, 24, 48])
    assert DEFAULT_WINDOWS in grid
    assert all(w.fast < w.medium < w.slow for w in grid)
