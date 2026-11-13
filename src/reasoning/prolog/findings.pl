/*  Layer 1 - Findings: numbers become clinical words.   (stratum 1)

    Format: r(RuleId, Stratum, (Head :- Body)).
    X is always a patient. Thresholds come from thresholds.pl.
*/

r(f_tachycardia, 1,
  (tachycardia(X) :- heart_rate(X, V), threshold(hr_high, T), V > T)).
r(f_severe_tachycardia, 1,
  (severe_tachycardia(X) :- heart_rate(X, V), threshold(hr_very_high, T), V > T)).

r(f_hypotension_map, 1,
  (hypotension(X) :- mean_arterial_pressure(X, V), threshold(map_low, T), V < T)).
r(f_hypotension_sbp, 1,
  (hypotension(X) :- systolic_bp(X, V), threshold(sbp_low, T), V < T)).
r(f_hypotension_flag, 1,
  (hypotension(X) :- low_blood_pressure(X))).

r(f_hypoxemia, 1,
  (hypoxemia(X) :- oxygen_saturation(X, V), threshold(spo2_low, T), V < T)).
r(f_severe_hypoxemia, 1,
  (severe_hypoxemia(X) :- oxygen_saturation(X, V), threshold(spo2_very_low, T), V < T)).

r(f_fever_temperature, 1,
  (has_fever(X) :- temperature(X, V), threshold(temp_fever, T), V >= T)).
r(f_fever_symptom, 1,
  (has_fever(X) :- has_symptom(X, fever))).

r(f_tachypnea, 1,
  (tachypnea(X) :- respiratory_rate(X, V), threshold(rr_high, T), V >= T)).
r(f_severe_tachypnea, 1,
  (severe_tachypnea(X) :- respiratory_rate(X, V), threshold(rr_very_high, T), V >= T)).

r(f_high_lactate, 1,
  (high_lactate(X) :- lactate(X, V), threshold(lactate_high, T), V >= T)).
r(f_very_high_lactate, 1,
  (very_high_lactate(X) :- lactate(X, V), threshold(lactate_very_high, T), V >= T)).

r(f_elderly, 1,
  (elderly(X) :- age(X, V), threshold(age_elderly, T), V >= T)).

r(f_altered_mental_state, 1,
  (altered_mental_state(X) :- has_symptom(X, confusion))).

% Closed-world assumption is unsafe for missing measurements: flag patients
% with neither core vitals nor any model prediction.
r(f_insufficient_data, 1,
  (insufficient_data(X) :- patient(X), \+ heart_rate(X, _), \+ oxygen_saturation(X, _),
                           \+ ml_risk(X, _, _), \+ dl_risk(X, _, _))).
