from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import optimization_page
from oscillator_optimization import VERSION, allocation_history, merge_saved, Windows


def configuration():
    return pd.DataFrame([{'Ticker': 'A', 'slow_dias': 144, 'medium_dias': 48,
                           'fast_dias': 24, 'versao': VERSION, 'fim_historico': '2014-01-01'}])


def history():
    rng = np.random.default_rng(10)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(.0002, .02, 1500))),
                     index=pd.bdate_range('2010-01-01', periods=1500))


def test_allocation_uses_saved_windows_and_keeps_missing_tickers():
    p = history()
    legacy = Mock()
    legacy.oscilador.return_value = pd.DataFrame({'pos_osc': [45., 70.]}, index=['A', 'B'])
    with patch.object(optimization_page, 'maximum_history', return_value=p), \
            patch.object(optimization_page, 'st'):
        result = optimization_page.allocation_with_saved_windows(
            legacy, pd.DataFrame({'A': p, 'B': p}), configuration())
    assert result.loc['A', 'pos_osc'] == allocation_history(p, Windows(144, 48, 24)).iloc[-1]
    assert result.loc['B', 'pos_osc'] == 70.


def test_provider_failure_keeps_original_allocation():
    legacy = Mock()
    original = pd.DataFrame({'pos_osc': [45.]}, index=['A'])
    legacy.oscilador.return_value = original
    with patch.object(optimization_page, 'maximum_history', side_effect=ValueError('offline')), \
            patch.object(optimization_page, 'st'):
        result = optimization_page.allocation_with_saved_windows(
            legacy, history().to_frame('A'), configuration())
    pd.testing.assert_frame_equal(result, original)


def test_historical_view_rejects_parameters_trained_in_its_future():
    legacy = Mock()
    original = pd.DataFrame({'pos_osc': [45.]}, index=['A'])
    legacy.oscilador.return_value = original
    with patch.object(optimization_page, 'maximum_history') as provider, \
            patch.object(optimization_page, 'st'):
        result = optimization_page.allocation_with_saved_windows(
            legacy, history().to_frame('A'), configuration(), as_of='2013-01-01')
    provider.assert_not_called()
    pd.testing.assert_frame_equal(result, original)


def test_maximum_history_requests_adjusted_full_history_without_padding():
    history_df = pd.DataFrame({'Adj Close': [100., 105., 106.]},
                              index=pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']))
    with patch.object(optimization_page.yf, 'Ticker') as ticker:
        ticker.return_value.history.return_value = history_df
        result = optimization_page.maximum_history.__wrapped__('^BVSP.SA')
    ticker.assert_called_once_with('^BVSP')
    args = ticker.return_value.history.call_args.kwargs
    assert args['period'] == 'max' and args['auto_adjust'] is False
    assert len(result) == 3


def test_page_run_review_and_save_without_external_writes():
    script = '''
import numpy as np
import pandas as pd
import streamlit as st
from unittest.mock import patch
import optimization_page
from oscillator_optimization import merge_saved
class FakeGoogle:
    def read_oscillator_windows(self):
        return st.session_state.get('fake_sheet', pd.DataFrame())
    def save_oscillator_windows(self, frame):
        st.session_state['fake_sheet'] = merge_saved(self.read_oscillator_windows(), frame)
rng = np.random.default_rng(10)
p = pd.Series(100 * np.exp(np.cumsum(rng.normal(.0002, .02, 1500))),
              index=pd.bdate_range('2010-01-01', periods=1500))
with patch.object(optimization_page, 'maximum_history', return_value=p):
    optimization_page.render_optimization_page(FakeGoogle(), ['A', 'B', 'Taxa_Juros_Brasil'])
'''
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert app.multiselect[0].options == ['A', 'B']
    app.button[0].click().run()
    assert not app.exception
    assert 'A' in app.session_state['oscillator_results']
    save = next(button for button in app.button if button.label == 'Salvar janelas no Position_Control')
    save.click().run()
    assert not app.exception
    assert app.session_state['fake_sheet'].Ticker.tolist() == ['A']


def test_google_persistence_only_writes_configuration_sheet():
    import plangoogle
    google = plangoogle.PlanGoogle.__new__(plangoogle.PlanGoogle)
    google.spreadsheetname = 'Position_Control'
    google.client = Mock()
    google.read_oscillator_windows = Mock(return_value=pd.DataFrame())
    with patch.object(plangoogle, 'Spread') as spread:
        google.save_oscillator_windows(configuration())
    call = spread.return_value.df_to_sheet.call_args
    assert call.kwargs['sheet'] == 'Oscillator_Windows'
    assert call.kwargs['index'] is False
    assert call.args[0].Ticker.tolist() == ['A']
