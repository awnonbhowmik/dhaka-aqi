"""
Dhaka AQI Analysis — source package.

Modules
-------
config       : shared constants, colour palettes, matplotlib rcParams
data_loader  : load and prepare the monthly dataset
analysis     : statistical analysis (trend, stationarity, effect sizes, health burden)
forecasting  : predictive models and multi-model ensemble (OLS, ETS, SARIMA, Prophet)
viz          : seaborn-first publication-quality figures
map_utils    : geospatial study area map (Dhaka City District)
"""

from .config      import PAL, SEASON_PAL, SEASON_ORDER, WHO, EPA, apply_style, POL_LATEX, POL_SHORT_LATEX
from .data_loader import load_data, compute_annual, compute_monthly_climatology
from .analysis    import (mann_kendall, run_mann_kendall_all, ols_trend_table,
                          run_adf_all, covid_effect_table, exceedance_summary,
                          health_burden, ekc_analysis, aqi_category_distribution,
                          descriptive_with_normality)
from .forecasting import (evaluate_models, forecast_to_2030,
                          scenario_projections, forecast_summary_table)
from .viz         import (save_fig,
                          plot_violin_by_season,
                          plot_interannual_boxplots,
                          plot_model_performance_heatmap)
from .map_utils   import plot_study_area

__all__ = [
    "PAL", "SEASON_PAL", "SEASON_ORDER", "WHO", "EPA", "apply_style",
    "POL_LATEX", "POL_SHORT_LATEX",
    "load_data", "compute_annual", "compute_monthly_climatology",
    "mann_kendall", "run_mann_kendall_all", "ols_trend_table",
    "run_adf_all", "covid_effect_table", "exceedance_summary",
    "health_burden", "ekc_analysis", "aqi_category_distribution",
    "descriptive_with_normality",
    "evaluate_models", "forecast_to_2030", "scenario_projections",
    "forecast_summary_table",
    "save_fig", "plot_study_area",
]
