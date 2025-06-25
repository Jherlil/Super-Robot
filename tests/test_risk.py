import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from risk import RiskManager


def test_can_trade_limits():
    rm = RiskManager(10, 2, 10, 3, 'normal', 2, 2, True, True, 0.8, ['EURUSD'])
    assert rm.can_trade('EURUSD')
    rm.assets['EURUSD']['losses_amount'] = 15
    assert not rm.can_trade('EURUSD')