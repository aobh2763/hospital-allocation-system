/*  Deduction by forward chaining.

    saturate/0 runs the rules stratum by stratum (1 = findings ... 12 = fallbacks).
    Inside a stratum it fires every rule whose body holds, stores each new
    head as a derived fact, and repeats until nothing new appears (fixpoint).
    Negation (\+) is only applied to predicates of lower strata, which are
    complete by then: the program is stratified, so the result is unique.

    Every derived fact remembers how it was obtained:
        derived(Fact, Seq, RuleId, Theta, Premises)
    Theta    = substitution found by unification, [[Var, Value], ...]
    Premises = [fact(G) | derived(G) | builtin(G) | naf(G)]
*/

:- dynamic derived/5.

saturate :-
    retractall(derived(_, _, _, _, _)),
    nb_setval(kb_seq, 0),
    findall(S, rule(_, S, _, _, _), Ss0),
    sort(Ss0, Strata),
    forall(member(S, Strata), saturate_stratum(S)).

saturate_stratum(S) :-
    fire_stratum(S, New),
    (   New > 0 -> saturate_stratum(S) ; true ).

fire_stratum(S, New) :-
    findall(x,
            ( rule(Id, S, H, B, Ns),
              solve(B, Premises),
              ground(H),
              \+ known(H),
              include([_ = V]>>nonvar(V), Ns, Bound),
              record(H, Id, Bound, Premises) ),
            L),
    length(L, New).

record(H, Id, Theta, Premises) :-
    nb_getval(kb_seq, N0),
    N is N0 + 1,
    nb_setval(kb_seq, N),
    assertz(derived(H, N, Id, Theta, Premises)).

% solve(+Body, -Premises): prove a rule body against the current facts.
solve(true, []) :- !.
solve((A, B), P) :- !,
    solve(A, PA), solve(B, PB), append(PA, PB, P).
solve((A ; B), P) :- !,
    (   solve(A, P) ; solve(B, P) ).
solve(\+ G, [naf(G)]) :- !,
    \+ solve(G, _).
solve(G, [builtin(G)]) :- is_builtin(G), !, call(G).
solve(G, [fact(G)]) :- is_base(G), !, call(G).
solve(G, [derived(G)]) :- derived(G, _, _, _, _).

% ---------------------------------------------------------------------------
% Reporting (called from Python)
% ---------------------------------------------------------------------------

kb_saturate(Count) :-
    saturate,
    aggregate_all(count, derived(_, _, _, _, _), Count).

% kb_deductions(-Steps): the full deduction log, in derivation order.
kb_deductions(Steps) :-
    findall(N-_{seq: N, fact: HS, rule: Id, stratum: S, theta: TS, premises: PS},
            ( derived(H, N, Id, Theta, Prem),
              rule(Id, S, _, _, _),
              fmt(H, HS),
              findall([V, VS], ( member(V = Val, Theta), fmt(Val, VS) ), TS),
              maplist(premise_dict, Prem, PS) ),
            Pairs),
    keysort(Pairs, Sorted),
    pairs_values(Sorted, Steps).

premise_dict(P, _{kind: K, term: S}) :-
    P =.. [K, G],
    fmt(G, S).

% kb_rule(+Id, -Dict): a rule as its author wrote it.
kb_rule(Id, _{id: Id, stratum: S, head: HS, body: BS}) :-
    rule(Id, S, H, B, Ns),
    fmt_vn(Ns, H, HS),
    conj_list(B, Lits),
    maplist(fmt_vn(Ns), Lits, BS).

kb_rules(L) :-
    findall(D, ( rule(Id, _, _, _, _), kb_rule(Id, D) ), L).

% kb_explain(+GoalString, -Tree): why a derived fact holds, as a tree of
% the rules and premises that produced it (deduction trace).
kb_explain(S, Tree) :-
    term_string(G, S),
    once(explain(G, Tree)).

explain(G, _{kind: derived, fact: GS, rule: Id, theta: TS, premises: Subs}) :-
    derived(G, _, Id, Theta, Prem), !,
    fmt(G, GS),
    findall([V, VS], ( member(V = Val, Theta), fmt(Val, VS) ), TS),
    maplist(explain_premise, Prem, Subs).
explain(G, _{kind: fact, fact: GS}) :-
    is_base(G), call(G), !,
    fmt(G, GS).
explain(G, _{kind: unknown, fact: GS}) :-
    fmt(G, GS).

explain_premise(derived(G), T) :- !, explain(G, T).
explain_premise(fact(G), _{kind: fact, fact: GS}) :- !, fmt(G, GS).
explain_premise(builtin(G), _{kind: builtin, fact: GS}) :- !, fmt(G, GS).
explain_premise(naf(G), _{kind: naf, fact: GS}) :- fmt(G, GS).
