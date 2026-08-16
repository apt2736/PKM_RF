#ifndef TRIAGE_PIPELINE_H
#define TRIAGE_PIPELINE_H
#ifdef __cplusplus
extern "C" {
#endif
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
typedef struct {
    float probs[5];
    int predicted_esi;
} TriageOutput;
TriageOutput predict_triage(const TriageInput* input);
#ifdef __cplusplus
}
#endif
#endif // TRIAGE_PIPELINE_H
