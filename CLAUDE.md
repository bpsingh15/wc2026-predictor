# CLAUDE.md — WC 2026 Predictor

This file is read by Claude Code on every session. It contains everything
needed to assist correctly without re-explanation.

---

## What this project is

A hybrid tabular + Graph Neural Network pipeline that predicts WC 2026
attacker performance (xG + xA per 90 minutes) using prior club-season stats
as features and past tournament performances as training labels.

Built to demonstrate: Azure Blob Storage, Databricks/PySpark, Graph Neural
Networks (PyTorch Geometric), OOP pipeline design, and proper temporal
cross-validation.

---

## Tech stack

- Python 3.11
- PySpark (Databricks Community Edition for cloud runs, local mode for dev)
- Azure Blob Storage via `azure-storage-blob`
- `soccerdata` for FBref scraping
- `xgboost` and `tabpfn` for tabular models
- `torch` + `torch-geometric` (GraphSAGE) for the GNN
- `click` for the CLI
- `pandas`, `scikit-learn`, `numpy`, `python-dotenv`

---

## Repository structure

```
wc2026-predictor/
  cli.py                   # Click entry point: train / predict commands
  config.py                # Loads env vars — no secrets in code
  pipeline/
    __init__.py
    ingestor.py            # Ingestor class
    transformer.py         # Transformer class (Spark)
    graph_builder.py       # GraphBuilder class (PyG)
    model_runner.py        # ModelRunner class + PlayerGNN model
    pipeline.py            # Pipeline orchestrator
  data/
    raw/                   # Local CSV cache — never re-scrape if exists here
    processed/             # Feature tables (Parquet)
  notebooks/               # Databricks exploration — never production logic
  tests/                   # pytest unit tests
  requirements.txt
  .env                     # Never committed — holds Azure connection strings
  README.md
```

---

## The four core classes

### Ingestor (`pipeline/ingestor.py`)
Pulls data from FBref via soccerdata and lands it in Azure Blob Storage.

Responsibilities:
- `ingest_club_season(league, season)` — scrape one league-season, cache locally, upload to Azure
- `ingest_tournament(tournament_name)` — same for tournament data
- `_already_cached(filename)` — check `data/raw/` before scraping
- `_upload_to_blob(filename)` — upload local file to Azure container

### Transformer (`pipeline/transformer.py`)
Spark pipeline that turns raw CSVs into a clean labeled feature table.

Responsibilities:
- `build_club_season_features(leagues, seasons)` — load CSVs from Azure, compute per-90s
- `build_feature_table()` — join club features to tournament labels on FBref player ID
- `save_feature_table(df, path)` — write Parquet to Azure Blob
- `validate_feature_table(df)` — print row counts, nulls, target distribution

### GraphBuilder (`pipeline/graph_builder.py`)
Constructs PyTorch Geometric Data objects from the feature table.

Responsibilities:
- `build_tournament_graph(exclude_tournament)` — build graph from all tournaments except one
- `_build_node_features(players_df)` — player stats → torch.Tensor
- `_build_edge_index(squad_df)` — construct bidirectional edge_index tensor

### ModelRunner (`pipeline/model_runner.py`)
Runs all models inside a LOTO CV loop and compares results.

Responsibilities:
- `loto_cv(model, df)` — leave-one-tournament-out cross-validation
- `run_xgboost_baseline()` — tabular only
- `run_gnn()` — GNN embeddings via GraphSAGE
- `run_hybrid()` — XGBoost on tabular + GNN embeddings concatenated
- `compare_models()` — returns results table with RMSE per fold

---

## Non-negotiable rules — always follow these

**1. Always check cache before scraping.**
Before any FBref call, check `_already_cached()`. If the file exists in
`data/raw/`, use it. Never re-scrape. FBref rate-limits aggressively.

**2. Never use random train/test splits.**
All validation is leave-one-tournament-out (LOTO). Random splits leak
temporal context. If you are about to write `train_test_split`, stop and
use the LOTO loop instead.

**3. Join on FBref player IDs, never player names.**
Player names cause mismatches (accents, abbreviations, transfers).
Always use the FBref player ID string as the join key.

**4. One class per file, one responsibility per method.**
No flat scripts except `cli.py`. No logic in `__init__` beyond
assigning parameters to `self`. If a method is longer than ~30 lines,
break it up.

**5. No hardcoded secrets or paths.**
Azure connection strings, account names, and container names always come
from environment variables. Load them via `python-dotenv` in `config.py`.
If you are about to write a connection string inline, stop.

**6. Add reverse edges in the graph.**
The player-team graph is undirected for message passing purposes.
Always concatenate `edge_index` with its flip:
`edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)`

**7. Spark for the full universe, pandas for models.**
Spark processes the raw club-season universe (all leagues, all players)
before filtering. After the feature table is built, load it into pandas
for XGBoost/GNN training. Don't force Spark into the model training loop.

---

## Data details

**Training tournaments:** WC2018, Euro2020, WC2022, Euro2024, CopaAmerica2024

**Feature columns (FEATURE_COLS):**
`xg_per90`, `npxg_per90`, `xag_per90`, `goals_per90`, `assists_per90`,
`key_passes_per90`, `shots_per90`, `progressive_passes_per90`,
`minutes` (durability), `age`, `position_encoded`,
`recent_intl_tournament_xg90` (null if no recent tournament)

**Target:** `xg_xa_per90` (tournament xG + xA per 90 minutes)

**Known data issues:**
- WC 2018 has sparse xG — handle as NaN (XGBoost handles natively)
- Copa America 2024 roster data may be incomplete
- Expected final row count: 200–400 player-tournament instances

**Azure Blob structure:**
```
wc2026-raw/
  raw/
    club_seasons/     # {league}_{season}_standard.csv
    tournaments/      # {tournament_name}_stats.csv
  processed/
    feature_table.parquet
```

---

## GNN architecture

Model: 2-layer GraphSAGE (`SAGEConv` from torch_geometric.nn)
- Layer 1: in_channels → 32, ReLU
- Layer 2: 32 → 16 (embeddings)

Node types:
- Attacking player nodes: features = FEATURE_COLS
- National team nodes: features = squad-averaged FEATURE_COLS

Edge types (all bidirectional):
- Player → National team ("plays for")
- Player → Player on same squad ("teammates")

Training: supervised on training-set player nodes using MSE loss against
`xg_xa_per90`. Extract final embeddings for held-out tournament players.

---

## Coding standards

- Type hints on all method signatures
- Docstring on every class and every public method
- Use `logging`, not `print()`, in production code
- PySpark: always import and use `F` from `pyspark.sql.functions`
- PyG `Data` objects: always set `x`, `edge_index`, `y`, and attach a
  `player_id_map` dict as a custom attribute for retrieving embeddings
- Test at minimum: `_already_cached`, the feature table join, `edge_index` shape

---

## Running the project

```bash
# Full training pipeline
python cli.py train

# Generate WC 2026 predictions
python cli.py predict data/wc2026_squads.csv

# Tests
pytest tests/ -v
```

---

## Current project status

[ ] Update this section as you progress through the build plan.

- [ ] Day 1: Environment setup
- [ ] Day 2: Azure Blob + Spark round-trip
- [ ] Day 3: Data discovery
- [ ] Day 4–5: Ingestor class
- [ ] Day 6–7: Transformer class
- [ ] Day 8: PyTorch Geometric tutorial
- [ ] Day 9: XGBoost baseline + LOTO CV
- [ ] Day 10: GraphBuilder
- [ ] Day 11: GNN model
- [ ] Day 12: Hybrid + comparison
- [ ] Day 13: CLI + Pipeline + 2026 predictions
- [ ] Day 14: README + Databricks deploy
