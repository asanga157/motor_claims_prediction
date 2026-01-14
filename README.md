_README_

How to run:

1) Install deps:
   pip install pandas numpy pyyaml

2) Run pipeline:
   python -m pipelines.data_pre_processing --input motor_repair_costs.csv --mode hard

Outputs:
- analysis/outputs/df_enriched.csv
- analysis/outputs/df_filtered_hard.csv or df_filtered_strict.csv
- analysis/outputs/report_missingness.csv
- analysis/outputs/report_flag_counts.csv
Logs:
- logs/pipeline.log
