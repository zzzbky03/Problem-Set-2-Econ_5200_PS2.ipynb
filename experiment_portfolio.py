"""Experiment-portfolio simulator and decision rules (ECON 5200 PS2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

NEAR_ZERO = 0.05

def simulate_portfolio(n: int, seed: int, base_rate: float = 0.10, p_real: float = 0.20,
                       mu_real: float = 0.50, sd_real: float = 0.20, sd_null: float = 0.02,
                       n_min: int = 2000, n_max: int = 100000,
                       planned_effect: float = 0.50) -> pd.DataFrame:
    """Return the experiment log with the true effect attached."""
    rng = np.random.default_rng(seed)
    is_real = rng.random(n) < p_real

    real_effects = rng.normal(mu_real, sd_real, n)
    null_effects = rng.normal(0, sd_null, n)
    true_effect = np.where(is_real, real_effects, null_effects)

    log_n = rng.uniform(np.log(n_min), np.log(n_max), n)
    n_per_arm = np.round(np.exp(log_n)).astype(int)

    conv_control = rng.binomial(n_per_arm, base_rate)
    conv_treat = rng.binomial(n_per_arm, base_rate + true_effect / 100)
    rate_control = conv_control / n_per_arm
    rate_treat = conv_treat / n_per_arm

    estimate = (rate_treat - rate_control) * 100
    se = np.sqrt(rate_control * (1 - rate_control) / n_per_arm
                 + rate_treat * (1 - rate_treat) / n_per_arm) * 100
    z = estimate / se
    p_value = 2 * stats.norm.sf(np.abs(z))

    se_plan = np.sqrt(2 * base_rate * (1 - base_rate) / n_per_arm) * 100
    power = stats.norm.sf(1.96 - planned_effect / se_plan) + stats.norm.cdf(-1.96 - planned_effect / se_plan)

    log = pd.DataFrame({
        'is_real': is_real,
        'true_effect': true_effect,
        'n_per_arm': n_per_arm,
        'estimate': estimate,
        'se': se,
        'z': z,
        'p_value': p_value,
        'power': power,
    })
    return log



def apply_rule(log: pd.DataFrame, rule: dict) -> pd.Series:
    """Ship decisions under `rule`: the p < 0.05 threshold and at least one other."""
    ship = log['p_value'] < rule['alpha']
    if rule['kind'] == 'power_floor':
        ship = ship & (log['power'] >= rule['min_power'])
    return ship

def evaluate(log: pd.DataFrame, rule: dict) -> dict:
    """Realised true return, ship count and false-ship rate under `rule`."""
    ship = apply_rule(log, rule)
    shipped = log[ship]
    n_shipped = len(shipped)
    if n_shipped == 0:
        false_rate = 0.0
    else:
        false_rate = float((shipped['true_effect'].abs() < NEAR_ZERO).mean())
    return {'realised_return': float(shipped['true_effect'].sum()),
            'n_shipped': n_shipped,
            'false_ship_rate': false_rate}


if __name__ == "__main__":
    log = simulate_portfolio(200, seed=42)
    client = {'kind': 'threshold', 'alpha': 0.05}
    floor = {'kind': 'power_floor', 'alpha': 0.05, 'min_power': 0.5}
    client_result = evaluate(log, client)
    floor_result = evaluate(log, floor)
    print('client rule:', client_result)
    print('power floor:', floor_result)

    assert len(log) == 200 and log['true_effect'].notna().all()
    assert floor_result['n_shipped'] <= client_result['n_shipped']
    assert 0 <= client_result['false_ship_rate'] <= 1 and 0 <= floor_result['false_ship_rate'] <= 1
    print('all assertions passed')   # demonstration plus at least three assertions that must hold
