/*  Layer 4 - Requirements: what each patient needs.   (strata 8 to 12)

    These predicates are the contract with Search (Block 4) and Planning
    (Block 5); see docs/predicates.md. Block 2 says what a patient needs and
    which resources could serve them. It never chooses the allocation.
*/

% ---- Patient needs (stratum 8) ----

r(q_urgent_evaluation, 8,
  (requires_urgent_evaluation(X) :- (high_risk(X) ; critical_risk(X)))).
r(q_reassess_disagreement, 8,
  (requires_reassessment(X) :- disagreement(X))).
r(q_reassess_low_confidence, 8,
  (requires_reassessment(X) :- low_confidence(X, _))).

r(q_icu, 8,                         % the PDF's rule: CriticalRisk(x) -> RequiresICU(x)
  (requires_icu(X) :- critical_risk(X))).
r(q_emergency_bed, 8,
  (requires_emergency_bed(X) :- (high_risk(X) ; medium_risk(X)))).
r(q_exam_room, 8,
  (requires_exam_room(X) :- low_risk(X))).

r(q_cardiology_chest_pain, 8,
  (candidate_specialist(X, cardiology) :- has_symptom(X, chest_pain), (high_risk(X) ; critical_risk(X)))).
r(q_cardiology_history, 8,
  (candidate_specialist(X, cardiology) :- has_condition(X, heart_disease), critical_risk(X))).
r(q_pulmonology, 8,
  (candidate_specialist(X, pulmonology) :- hypoxemia(X),
       (has_symptom(X, shortness_of_breath) ; has_condition(X, copd) ; has_condition(X, asthma)))).
r(q_general_surgery, 8,
  (candidate_specialist(X, general_surgery) :- has_symptom(X, abdominal_pain), (high_risk(X) ; critical_risk(X)))).
r(q_neurology, 8,
  (candidate_specialist(X, neurology) :- altered_mental_state(X), (high_risk(X) ; critical_risk(X)))).

r(q_nurse, 8,
  (requires_nurse(X) :- (high_risk(X) ; critical_risk(X)))).

r(q_ventilator, 8,
  (requires_equipment(X, ventilator) :- critical_risk(X), severe_hypoxemia(X))).
r(q_patient_monitor, 8,
  (requires_equipment(X, patient_monitor) :- (high_risk(X) ; critical_risk(X)))).
r(q_infusion_pump, 8,
  (requires_equipment(X, infusion_pump) :- hypotension(X))).
r(q_defibrillator, 8,
  (requires_equipment(X, defibrillator) :- has_symptom(X, chest_pain), critical_risk(X))).

r(q_monitoring, 8,
  (requires_monitoring(X, M) :- risk_level(X, L), monitoring_of(L, M))).

% ---- One primary specialist, bed type (stratum 9) ----

r(q_primary_specialist, 9,
  (requires_specialist(X, S) :- candidate_specialist(X, S), precedence(S, R),
                                \+ (candidate_specialist(X, S2), precedence(S2, R2), R2 < R))).

r(q_bed_icu,       9, (needs_bed_type(X, icu)       :- requires_icu(X))).
r(q_bed_emergency, 9, (needs_bed_type(X, emergency) :- requires_emergency_bed(X))).
r(q_bed_exam_room, 9, (needs_bed_type(X, exam_room) :- requires_exam_room(X))).

% ---- Doctor type (stratum 10) ----

r(q_general_doctor, 10,
  (requires_doctor(X, general) :- patient(X), \+ requires_specialist(X, _))).

% ---- Matching helpers: who could serve whom (stratum 11) ----
% Only waiting patients take part in the allocation problem.

r(m_can_treat_specialist, 11,
  (can_treat(D, X) :- waiting(X), requires_specialist(X, S), doctor(D), available(D), specialty(D, S))).
r(m_can_treat_general, 11,
  (can_treat(D, X) :- waiting(X), requires_doctor(X, general), doctor(D), available(D))).
r(m_suitable_bed, 11,
  (suitable_bed(B, X) :- waiting(X), needs_bed_type(X, T), bed(B), type(B, T), available(B))).
r(m_suitable_equipment, 11,
  (suitable_equipment(E, X) :- waiting(X), requires_equipment(X, T), equipment(E),
                               equipment_type(E, T), available(E))).
r(m_can_nurse, 11,
  (can_nurse(N, X) :- waiting(X), requires_nurse(X), nurse(N), available(N))).

% ---- Fallbacks when the ideal resource is missing (stratum 12) ----
% Search may use them, with a cost penalty.

r(m_fallback_doctor, 12,
  (fallback_doctor(D, X) :- waiting(X), requires_specialist(X, _), \+ can_treat(_, X),
                            doctor(D), available(D))).
r(m_fallback_bed, 12,
  (fallback_bed(B, X) :- waiting(X), requires_icu(X), \+ suitable_bed(_, X),
                         bed(B), type(B, emergency), available(B))).
