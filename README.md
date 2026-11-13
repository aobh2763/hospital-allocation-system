# Integrated intelligent agent for emergency management

GL4 Artificial Intelligence project, INSAT 2026. One agent that represents an
emergency department, learns risk from patient data, reasons with logic rules,
searches for a resource allocation, plans the actions, and adapts when the
situation changes. A teaching simulation, not a medical device.

## Setup (once)

1. SWI-Prolog: `sudo pacman -Syu swi-prolog` on Arch, or the installer from
   https://www.swi-prolog.org/download/stable.
2. Python packages, from this folder: `poetry install`
   (`janus-swi` compiles against SWI-Prolog, so install SWI-Prolog first).
3. Nothing else: the web interface is plain HTML/CSS/JS served by the Python
   server, so no Node.js is needed. PyTorch is only needed to retrain the DL
   model (`prediction/dl/export_model.py`); the agent runs the trained model with NumPy.

## Run the web interface

```bash
poetry run python src/web_server.py
```

Open http://127.0.0.1:8000.

- **Scenario tab (option 1):** the PDF's end-to-end example. "Run to next
  transfer", then "Play next event" three times (ICU1 lost, a second Critical
  patient, P2 deteriorating) with a few "Run 3" in between, and watch the agent
  re-decide in the right-hand panel.
- **Admit patient tab (option 2):** fill in the triage sheet (or "Load
  example") and "Admit and let the agent decide": the ML/DL estimates, the risk
  the rules derive and what the new allocation changes are shown at once.
- Click a bed, staff member or piece of equipment to take it out of service;
  click a patient for the deduction tree, the resolution proof, new vitals or discharge.

Options: `--given` uses the scenario's written predictions instead of the
models; `--port 8080`; `--scenario icu_reallocation.json`; `--frontend DIR`
serves another UI folder (default `front/frontend/`). The API is documented in
`docs/api.md`.

## Run in the terminal

```bash
poetry run python src/agent_demo.py        # option 1, end to end (--pause for a live demo)
poetry run python src/agent_cli.py         # option 2, add patients interactively
poetry run python src/patient_cli.py       # one patient: ML, DL, Prolog reasoning and proofs
poetry run python src/block2_demo.py       # Block 2 alone: deduction, resolution, unification
poetry run python src/block4_demo.py       # Block 4 alone: UCS vs A* on the scenarios
poetry run python src/search_compare.py    # UCS vs A* on 4 to 20 patients (CSV + figure)
poetry run python -m unittest discover -s tests
```

## Where things are

| Block | Code | Write-up |
| --- | --- | --- |
| 1 Environment | `src/patient.py`, `src/environment/` | `docs/block2_reasoning.md` (data sources) |
| 2 Reasoning | `src/reasoning/` (Prolog rules in `prolog/`) | `docs/block2_reasoning.md`, `docs/predicates.md` |
| 3 Learning | `prediction/` (ML notebooks, DL GRU), `src/reasoning/adapters.py` | `prediction/report/` |
| 4 Search | `src/search/` | `docs/block4_search.md` |
| 5 Planning | `src/planning/` | `docs/block5_planning.md` |
| 6 Agent | `src/agent/`, `src/web_server.py`, `front/frontend/` | `docs/block6_agent.md`, `docs/api.md` |
