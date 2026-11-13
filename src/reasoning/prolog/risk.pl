/*  Layer 3 - Risk: exactly one level per patient.   (strata 3 to 8)

    The final level is the highest level any source supports: in triage,
    under-estimating a patient is the worse error. Levels are settled from
    Critical down, so each rule only negates levels already complete.
*/

% Strongest opinion of each source, used to detect disagreement.
r(k_source_opinion, 3,
  (source_opinion(X, S, L) :- evidence(X, L, S), model_source(S), rank(L, R),
                              \+ (evidence(X, L2, S), rank(L2, R2), R2 > R))).

r(k_critical, 3,
  (critical_risk(X) :- evidence(X, critical, _))).
r(k_high, 4,
  (high_risk(X) :- evidence(X, high, _), \+ critical_risk(X))).
r(k_medium, 5,
  (medium_risk(X) :- evidence(X, medium, _), \+ high_risk(X), \+ critical_risk(X))).
r(k_low, 6,
  (low_risk(X) :- patient(X), \+ medium_risk(X), \+ high_risk(X), \+ critical_risk(X))).

r(k_level_critical, 7, (risk_level(X, critical) :- critical_risk(X))).
r(k_level_high,     7, (risk_level(X, high)     :- high_risk(X))).
r(k_level_medium,   7, (risk_level(X, medium)   :- medium_risk(X))).
r(k_level_low,      7, (risk_level(X, low)      :- low_risk(X))).

% Two sources two or more levels apart (e.g. ML says Low, DL says Critical).
r(k_disagreement, 7,
  (disagreement(X) :- source_opinion(X, S1, L1), source_opinion(X, S2, L2), S1 @< S2,
                      rank(L1, R1), rank(L2, R2), abs(R1 - R2) >= 2)).

r(k_priority, 8,
  (priority(X, N) :- risk_level(X, L), priority_of(L, N))).
r(k_max_wait, 8,
  (max_wait(X, M) :- risk_level(X, L), max_wait_of(L, M))).
