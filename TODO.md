ML Model for triage:

## Dataset path

datasets/5v_cleandf.RData

## Features:

- age
- gender
- triage_vital_hr
- triage_vital_rr
- triage_vital_sbp
- triage_vital_o2
- pulse_last
- resp_last
- spo2_last
- sbp_last

## Class

esi

## Structure

- Layer 1: Logistic Regression Model to decide ESI 1, ESI 5, or neither. If neither, continue to layer 2. On training, uses complete dataset
- Layer 2: Random Forest Model to decide ESI 2, ESI 3, ESI 4. On training, it uses dataset without rows containing esi 1 and 5

## Scoring

Uses Accuracy, ROC-AUC, Log Loss
