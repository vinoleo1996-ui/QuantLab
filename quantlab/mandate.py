"""Immutable account policy, owned by the operator rather than the forecasting model."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import json
from quantlab.config import Config
from quantlab.data.contracts import fingerprint


@dataclass(frozen=True)
class Mandate:
    frozen_config: str
    allowed_assets: tuple = ('stock', 'equity_etf')
    manual_recovery_required: bool = True

    @classmethod
    def from_config(cls, cfg):
        from quantlab.config_validation import validate_config
        validate_config(cfg)
        if cfg.capital.target_positions > 23 or cfg.capital.a_share_positions > 20:
            raise ValueError('mandate_position_limit')
        if cfg.research.max_stock_weight > .06 or cfg.research.max_etf_weight > .20:
            raise ValueError('mandate_security_limit')
        if cfg.research.cash_buffer<.02 or cfg.portfolio.max_sector_weight>.25 or cfg.research.max_cluster_weight>.30:
            raise ValueError('mandate_cash_or_group_limit')
        return cls(json.dumps(asdict(cfg),default=str,sort_keys=True))

    @property
    def version(self):
        policy=json.loads(self.frozen_config)
        policy.pop('data',None)  # Data location/retrieval is not an investment permission.
        return fingerprint(dict(policy=policy,allowed_assets=self.allowed_assets,
                                manual_recovery_required=self.manual_recovery_required))

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
