ML Model for triage:

## Dataset path

datasets/5v_cleandf.RData

## Features:

- age
- gender
- cc_breathingdifficulty
- triage_vital_hr
- triage_vital_sbp
- triage_vital_rr
- triage_vital_o2
- pulse_last
- resp_last
- spo2_last
- sbp_last
- pulse_min
- resp_min
- spo2_min
- sbp_min
- pulse_max
- resp_max
- spo2_max
- sbp_max

## Class

esi

## Feature Engineering

- is_dyspnea_total: if triage_vital_o2 < 90
- is_dyspnea_moderate: if triage_vital_o2 > 90 & triage_vital_o2 < 94
- is_bradypnea: if triage_vital_rr < 10
- is_tachypnea: if triage_vital_rr > 30
- is_hypotension: if triage_vital_sbp <= 90
- is_hypertension: if triage_vital_sbp > 220
- is_bradycardia_total: if triage_vital_hr < 40
- is_bradycardia_moderate: if triage_vital_hr > 40 & triage_vital_hr < 60
- is_tachycardia_total: if triage_vital_hr > 150
- is_tachycardia_moderate: if triage_vital_hr > 100 & triage_vital_hr < 150
- hr_mean_to_last: triage_vital_hr - pulse_last
- sbp_mean_to_last: triage_vital_sbp - sbp_last
- spo2_mean_to_last: triage_vital_o2 - spo2_last
- rr_mean_to_last: triage_vital_rr - resp_last
- hr_range: pulse_max - pulse_min
- rr_range: resp_max - resp_min
- spo2_range: spo2_max - spo2_min
- sbp_range: sbp_max - sbp_min
- hr_last_to_min: pulse_last - pulse_min
- sbp_last_to_min: resp_last - resp_min
- spo2_last_to_min: spo2_last - spo2_min
- sbp_last_to_min: sbp_last - sbp_min
- hr_last_to_max: pulse_last - pulse_max
- rr_last_to_max: resp_last - resp_max
- spo2_last_to_max: spo2_last - spo2_max
- sbp_last_to_max: sbp_last - sbp_max

## Structure

- Layer 1: XGBoost to handle intial class probabilities
- Layer 2: Logistic Regression receiving XGBoost input

## Scoring

Uses Accuracy, Precision, Recall, PR-AUC, ROC-AUC
