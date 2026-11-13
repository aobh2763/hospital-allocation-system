"""Conversion between Prolog notation and the PDF's notation.

Prolog reads capitalised words as variables, so the engine stores
``HighRisk(P1)`` as ``high_risk(p1)``. Everything shown to people goes through
:func:`to_pdf`, and everything people type goes through :func:`to_prolog`.

PDF notation conventions
------------------------
* predicates are CamelCase: ``HighRisk``, ``RequiresICU``, ``ML_Risk``
* constants are capitalised: ``P1``, ``ICU1``, ``Cardiology``, ``ChestPain``
* variables are a single lowercase letter, optionally followed by digits:
  ``x``, ``p``, ``x1``
"""

from __future__ import annotations

import re

# Words always written in capitals inside CamelCase names.
_ACRONYMS = {"icu", "ml", "dl", "bp", "copd", "ckd", "ct"}

# Predicates whose PDF name is not plain CamelCase.
_FUNCTOR_TO_PDF = {
    "ml_risk": "ML_Risk",
    "dl_risk": "DL_Risk",
    "mean_arterial_pressure": "MAP",
}
_FUNCTOR_TO_PROLOG = {v: k for k, v in _FUNCTOR_TO_PDF.items()}

# Constants kept lowercase in the PDF notation (evidence sources).
_LOWER_CONSTANTS = {"rules", "ml", "dl", "baseline"}

_ID_ATOM = re.compile(r"^[a-z]{1,4}\d+$")  # p1, icu1, eb2, d1 ...
_ID_PDF = re.compile(r"^[A-Z]{1,4}\d+$")  # P1, ICU1, EB2, D1 ...
_PDF_VAR = re.compile(r"^[a-z]\d*$")  # x, p, x1
_TOKEN = re.compile(
    r"""
    (?P<quoted>'(?:[^'\\]|\\.)*')
  | (?P<string>"(?:[^"\\]|\\.)*")
  | (?P<number>\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>\\\+|>=|=<|=:=|=\\=|\\==|\\=|==|@<|@>|:-|[(),;\[\]|+\-*/<>=])
  | (?P<space>\s+)
    """,
    re.VERBOSE,
)


def _camel(snake: str) -> str:
    return "".join(w.upper() if w in _ACRONYMS else w.capitalize() for w in snake.split("_") if w)


def _snake(camel: str) -> str:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", camel)
    return "_".join(p.lower() for p in parts)


def functor_to_pdf(name: str) -> str:
    return _FUNCTOR_TO_PDF.get(name, _camel(name))


def functor_to_prolog(name: str) -> str:
    return _FUNCTOR_TO_PROLOG.get(name, _snake(name))


def constant_to_pdf(atom: str) -> str:
    """``p1`` -> ``P1``, ``chest_pain`` -> ``ChestPain``, ``icu`` -> ``ICU``."""
    if _ID_ATOM.match(atom):
        return atom.upper()
    if atom in _LOWER_CONSTANTS:
        return atom
    return _camel(atom)


def to_atom(value: str) -> str:
    """Any user-facing name -> Prolog atom.

    ``P1`` -> ``p1``, ``ICU1`` -> ``icu1``, ``ChestPain`` -> ``chest_pain``,
    ``chest pain`` -> ``chest_pain``, ``General Surgery`` -> ``general_surgery``.
    """
    v = str(value).strip()
    if _ID_PDF.match(v):
        return v.lower()
    v = re.sub(r"[^A-Za-z0-9]+", "_", v).strip("_")
    if not v:
        raise ValueError(f"cannot turn {value!r} into a Prolog atom")
    atom = v.lower() if (v.islower() or v.isupper()) else _snake(v)
    if not re.match(r"^[a-z][a-z0-9_]*$", atom):
        raise ValueError(f"cannot turn {value!r} into a Prolog atom")
    return atom


def fact(name: str, *args) -> str:
    """Build a ground Prolog fact string, e.g. ``fact('age', 'P1', 72)``."""
    if not args:
        return name
    out = []
    for a in args:
        if isinstance(a, bool):
            raise TypeError("booleans are not fact arguments; use a predicate instead")
        if isinstance(a, (int, float)):
            out.append(repr(round(float(a), 4)) if isinstance(a, float) else str(a))
        else:
            out.append(to_atom(a))
    return f"{name}({','.join(out)})"


def to_pdf(text: str) -> str:
    """Rewrite a Prolog term (or list of terms) in the PDF notation."""
    tokens = [(m.lastgroup, m.group()) for m in _TOKEN.finditer(text)]
    out = []
    for i, (kind, tok) in enumerate(tokens):
        nxt = next((t for k, t in tokens[i + 1 :] if k != "space"), "")
        if kind == "ident":
            if nxt == "(":
                out.append(functor_to_pdf(tok))
            elif tok[0].isupper() or tok[0] == "_":
                out.append(_var_to_pdf(tok))
            else:
                out.append(constant_to_pdf(tok))
        elif kind == "quoted":
            out.append(tok[1:-1])
        elif kind == "op":
            out.append({"\\+": "¬", ";": " ∨ ", ">=": " ≥ ", "=<": " ≤ ", ">": " > ", "<": " < ",
                        ":-": " ← ", "=": " = ", "\\=": " ≠ "}.get(tok, tok))
        elif kind == "space":
            continue
        else:
            out.append(tok)
    s = "".join(out)
    s = re.sub(r",(?=\S)", ", ", s)
    return s


def _var_to_pdf(tok: str) -> str:
    if tok == "_":
        return "_"
    if tok.startswith("V_"):  # variable typed by a user in PDF notation
        return tok[2:]
    return tok.lower()


def to_prolog(text: str) -> str:
    """Rewrite PDF notation (``RequiresSpecialist(x, Cardiology)``) as Prolog."""
    tokens = [(m.lastgroup, m.group()) for m in _TOKEN.finditer(text)]
    out = []
    for i, (kind, tok) in enumerate(tokens):
        nxt = next((t for k, t in tokens[i + 1 :] if k != "space"), "")
        if kind == "ident":
            if nxt == "(":
                out.append(functor_to_prolog(tok))
            elif _PDF_VAR.match(tok):
                out.append("V_" + tok)
            elif tok == "_":
                out.append("_")
            else:
                out.append(to_atom(tok))
        elif kind == "space":
            continue
        else:
            out.append(tok)
    return "".join(out)
