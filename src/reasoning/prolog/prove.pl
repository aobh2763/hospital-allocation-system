/*  Resolution with a printed trace, failure explanations and a unification demo.

    kb_prove(+GoalString, -Result) answers a query such as "requires_icu(p1)"
    by SLD resolution, the refutation strategy Prolog itself uses: the negated
    goal is resolved against rules (clauses Head v not B1 v ... v not Bn) and
    facts until the empty clause is reached. Each step records

        goal       the literal being resolved
        clause     the rule or fact it is resolved with
        theta      the most general unifier found
        resolvent  the goals left afterwards ([] = empty clause, proof done)

    Negated literals (\+ G) are handled by negation as failure (SLDNF): the
    step succeeds when G has no proof.

    Resolution here works directly on rules and base facts. It does not use
    the derived/5 table, so it is an independent check of forward chaining.
*/

kb_prove(S, Result) :-
    term_string(G, S),
    (   sld([G], [], Steps)
    ->  Result = _{proved: true, steps: Steps}
    ;   why_not(G, 3, Why),
        Result = _{proved: false, steps: [], why: Why}
    ).

sld([], Acc, Steps) :- !, reverse(Acc, Steps).
sld([true | Gs], Acc, S) :- !, sld(Gs, Acc, S).
sld([(A, B) | Gs], Acc, S) :- !, sld([A, B | Gs], Acc, S).
sld([(A ; B) | Gs], Acc, S) :- !,
    (   sld([A | Gs], Acc, S) ; sld([B | Gs], Acc, S) ).
sld([\+ G | Gs], Acc, S) :- !,
    fmt(G, GS),
    \+ sld([G], [], _),
    resolvent(Gs, R),
    sld(Gs, [_{kind: naf, goal: GS, resolvent: R} | Acc], S).
sld([G | Gs], Acc, S) :-
    is_builtin(G), !,
    fmt(G, Before),
    call(G),
    fmt(G, After),
    resolvent(Gs, R),
    sld(Gs, [_{kind: builtin, goal: Before, result: After, resolvent: R} | Acc], S).
sld([G | Gs], Acc, S) :-
    is_base(G), !,
    goal_vars(G, Before, Named),
    call(G),
    fmt(G, FS),
    bound_named(Named, Theta),
    resolvent(Gs, R),
    sld(Gs, [_{kind: fact, goal: Before, clause: FS, theta: Theta, resolvent: R} | Acc], S).
sld([G | Gs], Acc, S) :-
    is_derived(G),
    goal_vars(G, Before, Named),
    rule(Id, _, H, B, Ns),
    fmt_vn(Ns, H, HS),
    conj_list(B, Lits),
    maplist(fmt_vn(Ns), Lits, BS),
    G = H,
    theta_pairs(Ns, RuleTheta),
    bound_named(Named, GoalTheta),
    append(GoalTheta, RuleTheta, Theta),
    resolvent([B | Gs], R),
    sld([B | Gs],
        [_{kind: rule, id: Id, goal: Before, head: HS, body: BS, theta: Theta, resolvent: R} | Acc],
        S).

% goal_vars(+G, -String, -Named): print G and remember which name each of its
% free variables was printed with, so theta can show what they became.
goal_vars(G, S, Named) :-
    term_variables(G, Vs),
    copy_term(G-Vs, C-Cs),
    numbervars(C, 0, _),
    with_output_to(string(S),
                   write_term(C, [quoted(true), numbervars(true), spacing(next_argument)])),
    maplist([V, CV, V-Name]>>with_output_to(string(Name),
                                            write_term(CV, [numbervars(true)])),
            Vs, Cs, Named).

bound_named(Named, Theta) :-
    findall([N, VS], ( member(V-N, Named), nonvar(V), fmt(V, VS) ), Theta).

resolvent(Gs, Strings) :-
    flatten_goals(Gs, Lits),
    copy_term(Lits, C),
    numbervars(C, 0, _),
    maplist([L, LS]>>with_output_to(string(LS),
                                    write_term(L, [quoted(true), numbervars(true),
                                                   spacing(next_argument)])),
            C, Strings).

flatten_goals([], []).
flatten_goals([true | Gs], L) :- !, flatten_goals(Gs, L).
flatten_goals([(A, B) | Gs], L) :- !, flatten_goals([A, B | Gs], L).
flatten_goals([G | Gs], [G | L]) :- flatten_goals(Gs, L).

% ---------------------------------------------------------------------------
% Why a query fails: for each rule whose head matches, the first body literal
% that cannot be proved, explained recursively up to Depth levels.
% ---------------------------------------------------------------------------

why_not(G, _, _{goal: GS, reason: "built-in test is false"}) :-
    is_builtin(G), !, fmt(G, GS).
why_not(G, _, _{goal: GS, reason: "no matching fact in the knowledge base"}) :-
    is_base(G), !, fmt(G, GS).
why_not(G, _, _{goal: GS, reason: "no rule or fact defines this predicate"}) :-
    \+ is_derived(G), !, fmt(G, GS).
why_not(G, Depth, _{goal: GS, reason: Reason, rules: Rules}) :-
    fmt(G, GS),
    findall(R, rule_failure(G, Depth, R), Rules),
    (   Rules == []
    ->  Reason = "no rule head unifies with this goal"
    ;   Reason = "every matching rule fails"
    ).

rule_failure(G, Depth, _{rule: Id, failed_at: FS, because: Sub}) :-
    copy_term(G, G1),
    rule(Id, _, H, B, _),
    G1 = H,
    conj_list(B, Lits),
    first_failure(Lits, Failed),
    fmt(Failed, FS),
    (   Depth > 1, Failed \= (\+ _), Failed \= (_ ; _)
    ->  D1 is Depth - 1, why_not(Failed, D1, Sub)
    ;   Failed = (\+ X)
    ->  fmt(X, XS),
        string_concat("negation fails because this is provable: ", XS, Msg),
        Sub = _{goal: FS, reason: Msg}
    ;   Sub = _{goal: FS, reason: "cannot be proved"}
    ).

first_failure([L | Ls], Failed) :-
    (   once(sld([L], [], _))
    ->  first_failure(Ls, Failed)
    ;   Failed = L
    ).

% ---------------------------------------------------------------------------
% Unification demo: kb_unify(+S1, +S2, -Result)
% ---------------------------------------------------------------------------

kb_unify(S1, S2, Result) :-
    % parsed together, so a variable name used in both terms is one variable
    format(string(Both), "u((~s), (~s))", [S1, S2]),
    term_string(u(T1, T2), Both, [variable_names(Ns)]),
    (   unify_with_occurs_check(T1, T2)
    ->  findall([N, VS], ( member(N = V, Ns), nonvar(V), fmt(V, VS) ), Theta),
        fmt(T1, US),
        Result = _{unified: true, theta: Theta, result: US}
    ;   clash(T1, T2, Kind, A, B),
        fmt(A, AS), fmt(B, BS),
        Result = _{unified: false, reason: Kind, a: AS, b: BS}
    ).

% clash(+T1, +T2, -Kind, -A, -B): the first pair of subterms that cannot unify.
% Kind is one of: occurs_check, constant_clash, constant_vs_compound, predicate_clash.
clash(A, B, occurs_check, V, T) :-
    (   var(A) ; var(B) ), !,
    (   var(A) -> V = A, T = B ; V = B, T = A ).
clash(A, B, constant_clash, A, B) :-
    atomic(A), atomic(B), !.
clash(A, B, constant_vs_compound, A, B) :-
    (   \+ compound(A) ; \+ compound(B) ), !.
clash(A, B, Kind, X, Y) :-
    functor(A, FA, NA), functor(B, FB, NB),
    (   FA/NA \== FB/NB
    ->  Kind = predicate_clash, X = FA/NA, Y = FB/NB
    ;   A =.. [_ | As], B =.. [_ | Bs],
        args_clash(As, Bs, Kind, X, Y)
    ).

args_clash([A | As], [B | Bs], Kind, X, Y) :-
    (   unify_with_occurs_check(A, B)
    ->  args_clash(As, Bs, Kind, X, Y)
    ;   clash(A, B, Kind, X, Y)
    ).
