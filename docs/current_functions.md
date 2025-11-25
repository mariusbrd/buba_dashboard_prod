# Funktionsinventar und Zielort

Dieses Dokument legt fest, in welcher der drei Dateien jede der aktuell vorhandenen Funktionen bleiben soll. Ziel ist es, bei der Refaktorierung keine Funktionalität zu verlieren. Es werden keine Funktionen in andere Dateien verschoben, sondern nur innerhalb der jeweiligen Datei neu geordnet oder in kleinere Teile zerlegt.

---

## 1. `forecast_integration.py`

**Zweck der Datei:** Adapter zwischen Dashboard und Kernpipeline. Nimmt Dashboard Daten an, bereitet sie für die Pipeline auf und formt das Ergebnis für die Anzeige.

### 1.1 Globale Hilfsfunktionen
- `_supports_utf8` → bleibt in `forecast_integration.py`
- `_sym` → bleibt in `forecast_integration.py`

### 1.2 Klasse `DashboardForecastAdapter`
Alle folgenden Methoden bleiben in `forecast_integration.py`:

- `__init__`
- `_load_exog_frame`
- `_to_ms`
- `_load_exog_from_default_cache`
- `_make_cache_tag`
- `_ui_target_to_excel_col`
- `_expand_exog_variants`
- `_exog_variants_for_name`
- `_has_all_requested_exogs`
- `_validate_exog_presence_in_excel`
- `_validate_exog_presence_in_df`
- `_get_file_stats`
- `_wait_for_loader_output`
- `_project_exog_series_monthly`
- `_extend_exogs_to_future_ms`
- `_calculate_confidence_intervals`
- `_add_confidence_intervals_to_forecast`
- `_cap_seasonal_quarter_lows`
- `prepare_pipeline_data`
- `create_temp_excel`
- `_build_config`
- `_extract_residuals_from_pipeline`
- `_generate_backtest_results`
- `_build_backtest_from_artifact`
- `run_forecast`
- `calculate_ci_coverage`
- `get_available_models`

**Hinweis:** In der Refaktorierung wird vor allem `run_forecast` in mehrere interne Schritte geteilt, bleibt aber in dieser Datei.

---

## 2. `forecaster_pipeline.py`

**Zweck der Datei:** Kernpipeline für Forecast. Einlesen, Aufbereiten, Featurebau, Modelltraining, Zukunftsaufbau, Modellverwaltung.

### 2.1 Globale Hilfsfunktionen
- `_supports_utf8` → bleibt in `forecaster_pipeline.py`
- `_sym` → bleibt in `forecaster_pipeline.py`

### 2.2 Klasse `Config`
- `Config` → bleibt in `forecaster_pipeline.py`
  - `ensure_paths`
  - `to_dict`

### 2.3 Klasse `ModelArtifact`
- `ModelArtifact` → bleibt in `forecaster_pipeline.py`
  - `__init__`
  - `save`
  - `load`
  - `exists`
  - `is_compatible`

### 2.4 Datei und Exog Verwaltung
- `get_model_filepath` → bleibt in `forecaster_pipeline.py`
- `harvest_exogs_from_downloader_output` → bleibt in `forecaster_pipeline.py`
- `autodetect_downloader_output` → bleibt in `forecaster_pipeline.py`
- `_to_jsonable` → bleibt in `forecaster_pipeline.py`
- `read_excel` → bleibt in `forecaster_pipeline.py`

### 2.5 Aufbereitung und Feature Engineering
- `aggregate_to_quarter` → bleibt in `forecaster_pipeline.py`
- `add_deterministic_features` → bleibt in `forecaster_pipeline.py`
- `month_lags_to_quarter_lags` → bleibt in `forecaster_pipeline.py`
- `build_quarterly_lags` → bleibt in `forecaster_pipeline.py`
- `_canonical_exog_name` → bleibt in `forecaster_pipeline.py`
- `resolve_exogs` → bleibt in `forecaster_pipeline.py`

### 2.6 Klasse `TargetYJ`
- `TargetYJ` → bleibt in `forecaster_pipeline.py`
  - `__init__`
  - `fit`
  - `transform`
  - `inverse`

### 2.7 Validierung und Metriken
- `expanding_splits` → bleibt in `forecaster_pipeline.py`
- `calculate_cv_metrics` → bleibt in `forecaster_pipeline.py`
- `calculate_insample_metrics` → bleibt in `forecaster_pipeline.py`
- `calculate_model_diagnostics` → bleibt in `forecaster_pipeline.py`
- `create_comprehensive_metadata` → bleibt in `forecaster_pipeline.py`

### 2.8 Extrapolation und Zukunftsdaten
- `extrapolate_with_arima` → bleibt in `forecaster_pipeline.py`
- `_fallback_extrapolation` → bleibt in `forecaster_pipeline.py`
- `_extrapolate_drift_seasonal` → bleibt in `forecaster_pipeline.py`
- `impute_future_exog_quarterly` → bleibt in `forecaster_pipeline.py`
- `build_future_design` → bleibt in `forecaster_pipeline.py`

### 2.9 Modelltraining und Prognose
- `train_best_model_h1` → bleibt in `forecaster_pipeline.py`
- `recursive_forecast` → bleibt in `forecaster_pipeline.py`
- `run_production_pipeline` → bleibt in `forecaster_pipeline.py`

*(In der Refaktorierung wird `run_production_pipeline` intern in vorbereiten, modell laden oder trainieren und prognose aufteilen, aber sie bleibt hier.)*

### 2.10 Modellverwaltung
- `list_saved_models` → bleibt in `forecaster_pipeline.py`
- `delete_model` → bleibt in `forecaster_pipeline.py`
- `compare_model_performance` → bleibt in `forecaster_pipeline.py`

---

## 3. `forecaster_main.py`

**Zweck der Datei:** Dash Ebene. Callbacks, Anzeige, Presets, Export.

### 3.1 Klasse
- `_CallbackProxy` → bleibt in `forecaster_main.py`
  - `__init__`
  - `callback`

### 3.2 Hilfsfunktionen für Daten aus dem Store
- `_filter_gvb_json_by_sektor` → bleibt in `forecaster_main.py`
- `_parse_store_df` → bleibt in `forecaster_main.py`
- `_make_metadata_jsonable` → bleibt in `forecaster_main.py`
- `_flatten_metadata_to_df` → bleibt in `forecaster_main.py`
- `_safe_load_store` → bleibt in `forecaster_main.py`
- `_normalize_dates` → bleibt in `forecaster_main.py`
- `_safe_series` → bleibt in `forecaster_main.py`
- `_slugify` → bleibt in `forecaster_main.py`
- `_current_quarter_end` → bleibt in `forecaster_main.py`
- `_looks_like_ecb_code` → bleibt in `forecaster_main.py`
- `_build_main_e1_table_from_store` → bleibt in `forecaster_main.py`
- `_find_ecb_db` → bleibt in `forecaster_main.py`
- `_load_ecb_options` → bleibt in `forecaster_main.py`
- `get_ecb_presets` → bleibt in `forecaster_main.py`
- `_load_ecb_series_names` → bleibt in `forecaster_main.py`
- `_load_hc_preset_cache` → bleibt in `forecaster_main.py`
- `_save_hc_preset_cache` → bleibt in `forecaster_main.py`
- `_normalize_target_slug` → bleibt in `forecaster_main.py`
- `_infer_target_from_slug_or_title` → bleibt in `forecaster_main.py`
- `_hydrate_hc_presets_with_cache` → bleibt in `forecaster_main.py`
- `get_ecb_presets_hydrated` → bleibt in `forecaster_main.py`
- `_write_final_dataset` → bleibt in `forecaster_main.py`
- `_snapshot_to_store_json` → bleibt in `forecaster_main.py`
- `_download_exog_codes` → bleibt in `forecaster_main.py`

### 3.3 Verarbeitung und Export der Exogenen
- `_merge_exogs_from_sources` → bleibt in `forecaster_main.py`
- `_extract_exog_list` → bleibt in `forecaster_main.py`
- `_make_export_bytes` → bleibt in `forecaster_main.py`

### 3.4 Darstellung von Backtests und Charts
- `_add_backtest_to_chart` → bleibt in `forecaster_main.py`
- `_add_backtest_error_band` → bleibt in `forecaster_main.py`
- `_add_backtest_markers` → bleibt in `forecaster_main.py`
- `create_backtest_controls` → bleibt in `forecaster_main.py`
- `_split_code_attrs` → bleibt in `forecaster_main.py`
- `_fmt_indicator_label` → bleibt in `forecaster_main.py`
- `_create_feature_importance` → bleibt in `forecaster_main.py`
- `create_feature_importance_icicle` → bleibt in `forecaster_main.py`
- `_empty_forecast_fig` → bleibt in `forecaster_main.py`
- `_error_forecast_response` → bleibt in `forecaster_main.py`
- `_create_pipeline_chart` → bleibt in `forecaster_main.py`

### 3.5 Optionen und Presets im Dashboard
- `build_exog_options` → bleibt in `forecaster_main.py`
- `notify_exog_add` → bleibt in `forecaster_main.py`
- `update_horizon_selection` → bleibt in `forecaster_main.py`
- `update_horizon_display` → bleibt in `forecaster_main.py`
- `populate_preset_dropdown_options` → bleibt in `forecaster_main.py`
- `toggle_load_preset_button` → bleibt in `forecaster_main.py`
- `apply_preset_to_model_store` → bleibt in `forecaster_main.py`
- `toggle_preset_modal` → bleibt in `forecaster_main.py`
- `save_preset_with_name` → bleibt in `forecaster_main.py`
- `toggle_delete_button` → bleibt in `forecaster_main.py`
- `toggle_delete_modal` → bleibt in `forecaster_main.py`
- `delete_selected_preset` → bleibt in `forecaster_main.py`
- `apply_preset_to_external_exog` → bleibt in `forecaster_main.py`
- `load_selected_preset` → bleibt in `forecaster_main.py`

### 3.6 Export, manuelle Reihen und Downloader
- `export_forecast_rawdata` → bleibt in `forecaster_main.py`
- `add_manual_series` → bleibt in `forecaster_main.py`
- `download_and_merge_exog` → bleibt in `forecaster_main.py`
- `prewarm_hc_presets` → bleibt in `forecaster_main.py`
- `show_initial_forecast_history` → bleibt in `forecaster_main.py`

### 3.7 Haupt Callback für die Prognose
- `_compute_simple_metrics` → bleibt in `forecaster_main.py`
- `create_pipeline_forecast` → bleibt in `forecaster_main.py`
- `toggle_backtest_visualization` → bleibt in `forecaster_main.py`
- `register_forecaster_callbacks` → bleibt in `forecaster_main.py`

---
