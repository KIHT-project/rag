from config import FILTER_EXCLUDED, FILTER_NOT_RELATED_TO_VTE, FILTER_REQUIRE_RISK_FACTORS, logger

def apply_filters(df):
    logger.info(
        "Applying filters [Excluded=%s, NotRelatedVTE=%s, RequireRiskFactors=%s]",
        FILTER_EXCLUDED, FILTER_NOT_RELATED_TO_VTE, FILTER_REQUIRE_RISK_FACTORS
    )
    total = len(df)
    mask = df.index.to_series().apply(lambda _: True)

    if FILTER_EXCLUDED and "Excluded" in df.columns:
        mask &= ~df["Excluded"]

    if FILTER_NOT_RELATED_TO_VTE and "Not related to venous thrombosis" in df.columns:
        mask &= ~df["Not related to venous thrombosis"]

    if FILTER_REQUIRE_RISK_FACTORS and "Reporting on risk factors" in df.columns:
        mask &= df["Reporting on risk factors"]

    filtered = df[mask]
    logger.info("Filtered %d -> %d", total, len(filtered))
    return filtered
