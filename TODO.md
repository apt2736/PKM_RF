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

## Structure

- Layer 1: Logistic Regression to decide if its ESI 1 or not. If not, continue to layer 2.
- Layer 2: Logistic Regression to decide if its ESI 5 or not, If not, continue to layer 3.
- Layer 3: Logistic Regression to decide if its ESI 2 or not. If not, continue to layer 4.
- Layer 4: Logistic Regression to decide if its ESI 3 or 4.

## Scoring

Uses Accuracy, Precision, Recall, PR-AUC, ROC-AUC
