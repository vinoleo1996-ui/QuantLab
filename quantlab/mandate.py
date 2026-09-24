"""Immutable account policy, owned by the operator rather than the forecasting model."""
from dataclasses import dataclass, asdict
import json
from quantlab.config import Config
from quantlab.data.contracts import fingerprint


@dataclass(frozen=True)
class Mandate:
    frozen_config: str
    allowed_assets: tuple = ('stock', 'equity_etf')
    manual_recovery_required: bool = True
    risk_policy: str = 'P0'

    @classmethod
    def from_config(cls, cfg, risk_policy='P0'):
        from quantlab.config_validation import validate_config
        from quantlab.strategy.risk_policy import PROFILES, policy_definition
        validate_config(cfg)
        policy_definition(risk_policy)  # Reject unknown permissions before checking bounds.
        envelope = (dict(max_stock_weight=.06, max_sector_weight=.25,
                         max_cluster_weight=.30, cash_buffer=.02)
                    if risk_policy == 'P0' else
                    PROFILES['normal' if risk_policy == 'P1' else 'active'])
        if cfg.capital.target_positions > 23 or cfg.capital.a_share_positions > 20:
            raise ValueError('mandate_position_limit')
        if cfg.research.max_stock_weight > envelope['max_stock_weight'] or cfg.research.max_etf_weight > .20:
            raise ValueError('mandate_security_limit')
        if (cfg.research.cash_buffer < envelope['cash_buffer'] or
            cfg.portfolio.max_sector_weight > envelope['max_sector_weight'] or
            cfg.research.max_cluster_weight > envelope['max_cluster_weight']):
            raise ValueError('mandate_cash_or_group_limit')
        if risk_policy != 'P0' and (cfg.portfolio.target_vol > envelope['target_vol'] or cfg.risk.drawdown_halt > .25):
            raise ValueError('mandate_risk_budget_limit')
        return cls(json.dumps(asdict(cfg),default=str,sort_keys=True), risk_policy=risk_policy)

    @property
    def version(self):
        policy=json.loads(self.frozen_config)
        policy.pop('data',None)  # Data location/retrieval is not an investment permission.
        identity = dict(policy=policy,allowed_assets=self.allowed_assets,
                        manual_recovery_required=self.manual_recovery_required)
        if self.risk_policy != 'P0':
            from quantlab.strategy.risk_policy import policy_definition
            identity['risk_policy'] = policy_definition(self.risk_policy)
        return fingerprint(identity)

    def config(self):
        from pathlib import Path
        cfg=Config()
        for group,values in json.loads(self.frozen_config).items():
            if isinstance(values,dict):
                for key,value in values.items():
                    if isinstance(getattr(getattr(cfg,group),key),Path): value=Path(value)
                    setattr(getattr(cfg,group),key,value)
            else: setattr(cfg,group,values)
        return cfg
