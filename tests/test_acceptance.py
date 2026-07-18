"""Scientific and reproducibility acceptance checks for the revised pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.aggregation.primary import build_daily, build_monthly
from src.aqi import aqi_category, pm25_aqi
from src.data_sources.airnow import parse_target_line
from src.forecasting import rolling_origin
from src.quality.completeness import day_coverage_pct, expected_days, is_complete_month

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/primary_observed_daily.parquet")


@pytest.fixture(scope="module")
def monthly() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data/processed/primary_observed_monthly.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_01_manifest_artifact_checksums_match() -> None:
    manifest = yaml.safe_load((ROOT / "data/source_manifest.yml").read_text())
    for source in manifest["sources"]:
        for artifact in source["artifacts"]:
            path = ROOT / artifact["filename"]
            assert path.stat().st_size == artifact["file_size"]
            assert sha256(path) == artifact["sha256"]


def test_02_daily_schema_has_required_lineage(daily: pd.DataFrame) -> None:
    required = {
        "timestamp_utc",
        "timestamp_local",
        "timezone",
        "date_local",
        "station_id",
        "pollutant",
        "value",
        "unit",
        "averaging_period",
        "provider",
        "original_provider",
        "instrument",
        "qa_flag",
        "source_file",
        "retrieval_date",
        "measurement_type",
    }
    assert required <= set(daily.columns)


def test_03_daily_key_is_unique(daily: pd.DataFrame) -> None:
    assert not daily.duplicated(["station_id", "date_local", "pollutant"]).any()


def test_04_selected_station_is_constant(daily: pd.DataFrame) -> None:
    assert set(daily["station_id"]) == {"DK1010001"}
    assert set(daily["station_name"]) == {"Dhaka"}


def test_05_physical_units_and_averaging_period_are_constant(daily: pd.DataFrame) -> None:
    assert set(daily["pollutant"]) == {"pm25"}
    assert set(daily["unit"]) == {"ug/m3"}
    assert set(daily["averaging_period"]) == {"24-hour"}


def test_06_timezone_and_utc_conversion_are_explicit(daily: pd.DataFrame) -> None:
    assert set(daily["timezone"]) == {"Asia/Dhaka"}
    local = pd.to_datetime(daily["timestamp_local"], utc=True)
    utc = pd.to_datetime(daily["timestamp_utc"], utc=True)
    assert local.equals(utc)


def test_07_values_are_finite_and_nonnegative(daily: pd.DataFrame) -> None:
    assert daily["value"].notna().all()
    assert (daily["value"] >= 0).all()


def test_08_calendar_completeness_functions() -> None:
    assert expected_days(2020, 2) == 29
    assert day_coverage_pct(23, 2020, 2) == pytest.approx(79.3103448)
    assert is_complete_month(23, 2020, 2)
    assert not is_complete_month(21, 2020, 2)


def test_09_monthly_coverage_is_recomputed(monthly: pd.DataFrame) -> None:
    calculated = monthly["valid_days"] / monthly["expected_days"] * 100
    assert calculated.to_numpy() == pytest.approx(monthly["day_coverage_pct"].to_numpy())
    assert monthly["is_complete"].eq(~monthly["is_partial"]).all()


def test_10_terminal_partial_month_is_excluded(monthly: pd.DataFrame) -> None:
    terminal = monthly.iloc[-1]
    assert terminal["month_start"] == "2025-03-01"
    assert terminal["valid_days"] == 24
    assert not terminal["is_complete"]
    analysis = pd.read_csv(ROOT / "data/processed/analysis_monthly.csv")
    assert len(analysis) == 71
    assert analysis["month_start"].max() == "2025-02-01"


@pytest.mark.parametrize(
    ("concentration", "expected"),
    [(0.0, 0), (9.0, 50), (9.1, 51), (35.4, 100), (35.5, 101),
     (55.4, 150), (55.5, 151), (125.4, 200), (125.5, 201),
     (225.4, 300), (225.5, 301), (325.4, 500), (400.0, 500)],
)
def test_11_epa_2024_aqi_boundaries(concentration: float, expected: int) -> None:
    assert pm25_aqi(concentration) == expected


def test_12_invalid_aqi_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        pm25_aqi(-0.1)


def test_13_aqi_categories_cover_boundaries() -> None:
    assert [aqi_category(value) for value in (50, 100, 150, 200, 300, 301)] == [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]


def test_14_dominant_pollutant_is_based_on_available_subindex(daily: pd.DataFrame) -> None:
    assert set(daily["dominant_pollutant"]) == {"pm25"}
    assert daily[["pm10_subindex", "no2_subindex", "so2_subindex", "co_subindex", "o3_subindex"]].isna().all().all()


def test_15_primary_series_contains_only_observed_ground_records(daily: pd.DataFrame) -> None:
    assert set(daily["measurement_type"]) == {"observed_ground"}
    assert not daily.astype(str).apply(lambda column: column.str.contains("modeled|reconstructed", case=False).any()).any()


def test_16_source_url_date_matches_each_observation(daily: pd.DataFrame) -> None:
    url_dates = daily["source_file"].str.extract(r"/(\d{8})/daily_data_v2\.dat$")[0]
    assert url_dates.notna().all()
    assert (pd.to_datetime(url_dates).dt.strftime("%Y-%m-%d") == daily["date_local"]).all()
    assert (pd.to_datetime(daily["date_local"]) <= pd.to_datetime(daily["retrieval_date"])).all()


def test_17_no_month_is_a_single_value_repeated_daily(daily: pd.DataFrame) -> None:
    month = pd.to_datetime(daily["date_local"]).dt.to_period("M")
    assert daily.groupby(month)["value"].nunique().gt(1).all()


def test_18_legacy_values_are_all_excluded() -> None:
    ledger = pd.read_csv(ROOT / "data/provenance/legacy_observation_provenance.csv")
    assert set(ledger["verification_status"]) <= {"classified_excluded", "unknown_excluded"}
    assert ledger["verification_status"].str.endswith("excluded").all()
    assert ledger["exclusion_reason"].notna().all()


def test_19_forecast_is_positive_and_interval_ordered() -> None:
    forecast = pd.read_csv(ROOT / "tables/forecast_summary.csv")
    assert len(forecast) == 24
    assert (forecast[["forecast", "lower_95", "upper_95"]].to_numpy() >= 0).all()
    assert (forecast["lower_95"] <= forecast["forecast"]).all()
    assert (forecast["forecast"] <= forecast["upper_95"]).all()


def test_20_forecast_coverage_is_a_percentage() -> None:
    ranking = pd.read_csv(ROOT / "tables/model_ranking.csv")
    assert ranking["interval_coverage_pct"].between(0, 100).all()
    assert ranking.iloc[0]["model"] == "sarima"


def test_21_scenarios_are_not_mislabeled_as_forecasts(daily: pd.DataFrame) -> None:
    scenario = pd.read_csv(ROOT / "tables/scenario_summary.csv")
    assert scenario[["lower_95", "upper_95"]].isna().all().all()
    assert not scenario["type"].str.contains("forecast", case=False).any()
    dates = pd.to_datetime(daily["date_local"])
    observed_2024 = daily.loc[dates.dt.year == 2024, "value"].mean()
    assert scenario.iloc[0]["pm25_ug_m3"] == pytest.approx(observed_2024)


def test_22_averaging_periods_are_not_mixed() -> None:
    annual = pd.read_csv(ROOT / "tables/annual_guideline_comparison.csv")
    daily_exceedance = pd.read_csv(ROOT / "tables/daily_exceedance_summary.csv")
    hourly = pd.read_csv(ROOT / "tables/hourly_exceedance_summary.csv")
    assert set(annual["averaging_period"]) == {"annual"}
    assert daily_exceedance["averaging_period"].str.startswith("24-hour").all()
    assert hourly.empty


def test_23_manuscript_key_values_match_generated_tables() -> None:
    paper = (ROOT / "paper/revised/paper_revised.md").read_text()
    descriptive = pd.read_csv(ROOT / "tables/descriptive_summary.csv").iloc[0]
    assert f"{descriptive['mean_ug_m3']:.2f} ug/m3" in paper
    assert "71 complete months" in paper
    assert "p=0.0046" in paper
    assert "24-month forecast" in paper


def test_24_every_manuscript_figure_exists() -> None:
    paper = (ROOT / "paper/revised/paper_revised.md").read_text()
    paths = re.findall(r"\]\((figures/[^)]+)\)", paper)
    assert paths
    assert all((ROOT / path).is_file() for path in paths)


def test_25_readme_dates_match_analysis_products() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "2019-01 through 2025-02" in readme
    assert "March 2025 is partial" in readme


def test_26_validation_is_separate_and_not_fabricated() -> None:
    validation = pd.read_csv(ROOT / "data/processed/validation_observed_monthly.csv")
    assert validation.empty
    assert "validation" not in set(
        pd.read_parquet(ROOT / "data/processed/primary_observed_daily.parquet")["measurement_type"]
    )


def test_27_airnow_parser_preserves_lineage() -> None:
    line = (
        "01/02/19|DK1010001|Dhaka|PM2.5-24hr|UG/M3|42.1|24|"
        "U.S. Department of State Bangladesh - Dhaka|118|3|23.796374|90.424614|"
        "050DK1010001|"
    )
    record = parse_target_line(line, "https://example.test/20190102.dat", "2026-01-01T00:00:00+00:00", "abc")
    assert record is not None
    assert record.date_local == "2019-01-02"
    assert record.source_line == line
    assert record.response_sha256 == "abc"


def test_28_clean_fixture_builds_without_imputation(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "date_local": ["2024-01-01", "2024-01-02"],
            "station_id": ["DK1010001"] * 2,
            "station_name": ["Dhaka"] * 2,
            "parameter": ["PM2.5-24hr"] * 2,
            "unit": ["UG/M3"] * 2,
            "value": [10.0, 20.0],
            "duration_hours": [24] * 2,
            "agency": ["U.S. Department of State Bangladesh - Dhaka"] * 2,
            "source_aqi": [52, 68],
            "latitude": [23.796374] * 2,
            "longitude": [90.424614] * 2,
            "source_url": ["https://example.test/day.dat"] * 2,
            "retrieval_timestamp_utc": ["2026-01-01T00:00:00+00:00"] * 2,
        }
    )
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)
    built_daily = build_daily(path)
    built_monthly = build_monthly(built_daily)
    assert built_daily["value"].tolist() == [10.0, 20.0]
    assert built_monthly.iloc[0]["pm25_mean"] == 15.0
    assert not built_monthly.iloc[0]["is_complete"]


def test_29_cams_never_enters_observed_product(daily: pd.DataFrame) -> None:
    assert not daily["source_id"].str.contains("cams", case=False).any()
    assert not daily["provider"].str.contains("cams", case=False).any()


def test_30_coverage_uses_zero_to_one_hundred_scale(monthly: pd.DataFrame) -> None:
    assert monthly["day_coverage_pct"].between(0, 100).all()
    assert monthly["day_coverage_pct"].max() > 1


def test_31_rolling_origin_evaluation_is_deterministic() -> None:
    dates = pd.Series(pd.date_range("2019-01-01", periods=48, freq="MS"))
    values = 80 + 30 * np.cos(2 * np.pi * np.arange(48) / 12) + np.arange(48) * 0.2
    first = rolling_origin(values, dates, horizon=12)
    second = rolling_origin(values, dates, horizon=12)
    pd.testing.assert_frame_equal(first, second)
