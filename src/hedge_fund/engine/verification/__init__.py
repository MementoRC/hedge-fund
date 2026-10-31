"""Thesis and rating verification engine (pure): score a researched name against its excess
return over the benchmark and summarise the observations per horizon.
"""

from hedge_fund.engine.verification.observe import observe
from hedge_fund.engine.verification.stats import make_bucket, score_key, spearman

__all__ = ["make_bucket", "observe", "score_key", "spearman"]
