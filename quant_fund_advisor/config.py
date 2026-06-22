"""Default sector mappings and investable fund universe."""

US_SECTOR_ETFS = {
    "technology": "XLK",
    "semiconductor": "SOXX",
    "communication": "XLC",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "healthcare": "XLV",
    "financials": "XLF",
    "industrials": "XLI",
    "materials": "XLB",
    "energy": "XLE",
    "utilities": "XLU",
    "real_estate": "XLRE",
}

# Liquid A-share ETF proxies. Codes are examples, not a recommendation list.
A_SHARE_SECTOR_ETFS = {
    "technology": "159995",
    "semiconductor": "512480",
    "communication": "515880",
    "consumer_discretionary": "159928",
    "consumer_staples": "512690",
    "healthcare": "512010",
    "financials": "512800",
    "industrials": "516800",
    "materials": "512400",
    "energy": "159930",
    "utilities": "159611",
    "real_estate": "512200",
}

DEFAULT_FUND_UNIVERSE_COLUMNS = [
    "fund_code",
    "fund_name",
    "sector",
    "purchase_fee",
    "redemption_fee",
    "min_holding_days",
    "available_on_alipay",
]

