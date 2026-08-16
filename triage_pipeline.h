#ifndef TRIAGE_PIPELINE_H
#define TRIAGE_PIPELINE_H

#ifdef __cplusplus
extern "C" {
#endif

// Raw Input Features (15 Values)
typedef struct {
    float age;
    float cc_breathingdifficulty;
    float gender;
    float triage_vital_hr;
    float triage_vital_sbp;
    float triage_vital_rr;
    float triage_vital_o2;
    float pulse_min;
    float resp_min;
    float spo2_min;
    float sbp_min;
    float pulse_max;
    float resp_max;
    float spo2_max;
    float sbp_max;
} TriageInput;
// 5-Class Output Probabilities and Predicted ESI Level
typedef struct {
    float probs[5]; // Index 0..4 maps to ESI 1..5
    int predicted_esi; // 1..5
} TriageOutput;
// Primary C Pipeline Entry Point
TriageOutput predict_triage(const TriageInput* input);
#ifdef __cplusplus
}
#endif
#endif // TRIAGE_PIPELINE_H
