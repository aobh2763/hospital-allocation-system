/*  Knowledge Base core: base facts, rule store and the shared solver.

    Base facts are ground atoms written by Python (patients, predictions,
    resources). Rules are read from the layer files as terms of the form

        r(Id, Stratum, (Head :- Body)).

    and stored as rule(Id, Stratum, Head, Body, VariableNames), so traces can
    print the rule with the variable names the author wrote (X, V, T, ...).

    Derived facts are only ever produced by forward chaining (forward.pl) and
    stored as derived(Fact, Seq, RuleId, Theta, Premises).
*/

:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(yall)).

:- dynamic rule/5.
:- dynamic derived/5.

% ---------------------------------------------------------------------------
% Base predicates: the only facts Python may assert (see docs/predicates.md).
% ---------------------------------------------------------------------------

base_predicate(patient/1).
base_predicate(age/2).
base_predicate(sex/2).
base_predicate(heart_rate/2).
base_predicate(systolic_bp/2).
base_predicate(diastolic_bp/2).
base_predicate(mean_arterial_pressure/2).
base_predicate(oxygen_saturation/2).
base_predicate(temperature/2).
base_predicate(respiratory_rate/2).
base_predicate(lactate/2).
base_predicate(has_symptom/2).
base_predicate(has_condition/2).
base_predicate(low_blood_pressure/1).
base_predicate(waiting/1).
base_predicate(under_treatment/1).
base_predicate(ml_risk/3).
base_predicate(dl_risk/3).
base_predicate(doctor/1).
base_predicate(specialty/2).
base_predicate(nurse/1).
base_predicate(bed/1).
base_predicate(type/2).
base_predicate(equipment/1).
base_predicate(equipment_type/2).
base_predicate(available/1).
base_predicate(assigned/2).
base_predicate(surge_day/0).
base_predicate(diversion/0).

% Static lookup tables (thresholds.pl). Treated like base facts by the solver.
table_predicate(threshold/2).
table_predicate(rank/2).
table_predicate(priority_of/2).
table_predicate(max_wait_of/2).
table_predicate(monitoring_of/2).
table_predicate(precedence/2).
table_predicate(model_source/1).

:- forall(base_predicate(N/A), dynamic(N/A)).
:- dynamic threshold/2.

is_base(G) :-
    functor(G, N, A),
    ( base_predicate(N/A) ; table_predicate(N/A) ), !.

is_builtin(G) :-
    functor(G, N, A),
    memberchk(N/A, [ (>)/2, (<)/2, (>=)/2, (=<)/2, (=:=)/2, (=\=)/2,
                     (=)/2, (\=)/2, (==)/2, (\==)/2, (@<)/2, (@>)/2, (is)/2 ]), !.

is_derived(G) :-
    functor(G, N, A),
    functor(H, N, A),
    rule(_, _, H, _, _), !.

% known(+Goal): Goal holds as a base fact, a table entry or a derived fact.
known(G) :- is_base(G), !, call(G).
known(G) :- derived(G, _, _, _, _).

% ---------------------------------------------------------------------------
% Loading rules
% ---------------------------------------------------------------------------

clear_rules :- retractall(rule(_, _, _, _, _)).

load_rules(File) :-
    setup_call_cleanup(open(File, read, In), read_rules(In), close(In)).

read_rules(In) :-
    read_term(In, T, [variable_names(Ns)]),
    (   T == end_of_file
    ->  true
    ;   add_rule(T, Ns),
        read_rules(In)
    ).

add_rule(r(Id, Stratum, (H :- B)), Ns) :- !,
    check_new_rule(Id),
    assertz(rule(Id, Stratum, H, B, Ns)).
add_rule(r(Id, Stratum, H), Ns) :- !,
    check_new_rule(Id),
    assertz(rule(Id, Stratum, H, true, Ns)).
add_rule(T, _) :-
    throw(error(domain_error(rule_term, T), _)).

check_new_rule(Id) :-
    (   rule(Id, _, _, _, _)
    ->  throw(error(permission_error(redefine, rule, Id), _))
    ;   true
    ).

% ---------------------------------------------------------------------------
% Base facts (called from Python with term strings)
% ---------------------------------------------------------------------------

kb_add(S) :-
    term_string(T, S),
    assert_base(T).

assert_base(T) :-
    (   \+ is_base(T)
    ->  throw(error(domain_error(base_fact, T), _))
    ;   \+ ground(T)
    ->  throw(error(instantiation_error, T))
    ;   call(T)
    ->  true                                    % no duplicates
    ;   assertz(T)
    ).

% kb_remove(+PatternString): retract every base fact matching the pattern,
% for example "heart_rate(p1,_)" or "available(icu1)".
kb_remove(S) :-
    term_string(T, S),
    (   is_base(T) -> retractall(T) ; throw(error(domain_error(base_fact, T), _)) ).

% kb_remove_patient(+P): forget everything about patient P.
kb_remove_patient(P0) :-
    atom_string(P, P0),                         % Python passes strings
    forall(( base_predicate(N/A), A >= 1 ),
           ( functor(H, N, A), arg(1, H, P), retractall(H) )).

kb_clear :-
    forall(base_predicate(N/A), ( functor(H, N, A), retractall(H) )),
    retractall(derived(_, _, _, _, _)).

kb_set_threshold(Name0, Value) :-
    atom_string(Name, Name0),
    retractall(threshold(Name, _)),
    assertz(threshold(Name, Value)).

% kb_base_facts(-Strings): every base fact, for display.
kb_base_facts(L) :-
    findall(S,
            ( base_predicate(N/A), functor(H, N, A), call(H), fmt(H, S) ),
            L).

% kb_solutions(+PatternString, -Rows): all bindings of the pattern's variables
% over base + derived facts, each row a list of [Name, Value] pairs.
kb_solutions(S, Rows) :-
    term_string(T, S, [variable_names(Ns)]),
    findall(Row, ( known(T), maplist(binding_pair, Ns, Row) ), Rows0),
    sort(Rows0, Rows).

binding_pair(N = V, [N, Out]) :-
    (   atomic(V) -> Out = V ; fmt(V, Out) ).

% ---------------------------------------------------------------------------
% Formatting helpers shared by forward.pl and prove.pl
% ---------------------------------------------------------------------------

% fmt(+Term, -String): print a term, naming free variables A, B, C ...
fmt(T, S) :-
    copy_term(T, C),
    numbervars(C, 0, _),
    with_output_to(string(S),
                   write_term(C, [quoted(true), numbervars(true), spacing(next_argument)])).

% fmt_vn(+VariableNames, +Term, -String): print with the author's variable names.
% Variables the author left anonymous print as _.
fmt_vn(Ns, T, S) :-
    copy_term(Ns-T, Ns1-T1),
    maplist([N = V]>>( var(V) -> V = '$VAR'(N) ; true ), Ns1),
    term_variables(T1, Rest),
    maplist(=('$VAR'('_')), Rest),
    with_output_to(string(S),
                   write_term(T1, [quoted(true), numbervars(true), spacing(next_argument)])).

conj_list(true, []) :- !.
conj_list((A, B), L) :- !, conj_list(A, LA), conj_list(B, LB), append(LA, LB, L).
conj_list(G, [G]).

theta_pairs(Ns, Pairs) :-
    findall([N, VS], ( member(N = V, Ns), nonvar(V), fmt(V, VS) ), Pairs).
