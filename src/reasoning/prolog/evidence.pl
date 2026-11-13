/*  Layer 2 - Evidence: what each source says about the risk.   (stratum 2)

    evidence(X, Level, Source), Source in {rules, ml, dl, baseline}.
    This is where Machine Learning and Deep Learning join the reasoning:
    a confident prediction is one more piece of evidence.
*/

% ---- Explicit clinical knowledge (source = rules) ----

r(e_shock, 2,
  (evidence(X, critical, rules) :- hypotension(X), severe_hypoxemia(X))).
r(e_septic_shock, 2,
  (evidence(X, critical, rules) :- hypotension(X), very_high_lactate(X))).

r(e_chest_pain_hypotension, 2,      % the PDF's rule: ChestPain AND LowBloodPressure -> HighRisk
  (evidence(X, high, rules) :- has_symptom(X, chest_pain), hypotension(X))).
r(e_tachypnea_hypotension, 2,
  (evidence(X, high, rules) :- tachypnea(X), hypotension(X))).
r(e_tachypnea_confusion, 2,
  (evidence(X, high, rules) :- tachypnea(X), altered_mental_state(X))).
r(e_elderly_dyspnea, 2,
  (evidence(X, high, rules) :- elderly(X), hypoxemia(X), has_symptom(X, shortness_of_breath))).
r(e_severe_vital_sign, 2,
  (evidence(X, high, rules) :- (severe_tachycardia(X) ; severe_tachypnea(X) ; severe_hypoxemia(X)))).

r(e_fever_tachycardia, 2,
  (evidence(X, medium, rules) :- has_fever(X), tachycardia(X))).
r(e_hypoxemia, 2,
  (evidence(X, medium, rules) :- hypoxemia(X))).
r(e_chest_pain, 2,
  (evidence(X, medium, rules) :- has_symptom(X, chest_pain))).
r(e_high_lactate, 2,
  (evidence(X, medium, rules) :- high_lactate(X))).

r(e_baseline, 2,
  (evidence(X, low, baseline) :- patient(X))).

% ---- Learned evidence (Block 3) ----

r(e_ml, 2,
  (evidence(X, L, ml) :- ml_risk(X, L, P), threshold(model_confidence, C), P >= C)).
r(e_dl, 2,
  (evidence(X, L, dl) :- dl_risk(X, L, P), threshold(model_confidence, C), P >= C)).

r(e_ml_low_confidence, 2,
  (low_confidence(X, ml) :- ml_risk(X, _, P), threshold(model_confidence, C), P < C)).
r(e_dl_low_confidence, 2,
  (low_confidence(X, dl) :- dl_risk(X, _, P), threshold(model_confidence, C), P < C)).
