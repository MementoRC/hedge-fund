"""Shared view layer for the dashboard and the site.

``load`` is the only module here that touches adapters; this package re-exports nothing from
it, so importing ``hedge_fund.viz`` never pulls an adapter in.
"""
