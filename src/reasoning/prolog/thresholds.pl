/*  Numeric thresholds and lookup tables used by the rules.

    Educational approximations inspired by NEWS2 and qSOFA. This project is a
    simulation and none of these numbers is medical advice. Tune them here
    (or from Python with KnowledgeBase.set_threshold) without touching rules.
*/

:- dynamic threshold/2.

threshold(hr_high,            110).    % bpm, tachycardia
threshold(hr_very_high,       130).    % bpm, severe tachycardia
threshold(map_low,             65).    % mmHg, hypotension (mean arterial pressure)
threshold(sbp_low,             90).    % mmHg, hypotension (systolic)
threshold(spo2_low,            92).    % %, hypoxemia
threshold(spo2_very_low,       88).    % %, severe hypoxemia
threshold(temp_fever,        38.0).    % degrees C
threshold(rr_high,             22).    % breaths/min, tachypnea
threshold(rr_very_high,        30).    % breaths/min, severe tachypnea
threshold(lactate_high,       2.0).    % mmol/L
threshold(lactate_very_high,  4.0).    % mmol/L
threshold(age_elderly,         65).    % years
threshold(model_confidence,  0.60).    % minimum probability for ML/DL evidence

% Severity order of risk levels.
rank(low, 1).
rank(medium, 2).
rank(high, 3).
rank(critical, 4).

% Sources whose opinions are compared for disagreement (not the baseline).
model_source(rules).
model_source(ml).
model_source(dl).

% Search inputs: priority (1 = first) and longest acceptable wait in minutes,
% on a Manchester-style triage scale.
priority_of(critical, 1).
priority_of(high, 2).
priority_of(medium, 3).
priority_of(low, 4).

max_wait_of(critical, 0).
max_wait_of(high, 10).
max_wait_of(medium, 60).
max_wait_of(low, 120).

monitoring_of(critical, continuous).
monitoring_of(high, hourly).
monitoring_of(medium, routine).
monitoring_of(low, routine).

% When several specialists match, the lowest number wins (one primary only).
precedence(cardiology, 1).
precedence(neurology, 2).
precedence(general_surgery, 3).
precedence(pulmonology, 4).
