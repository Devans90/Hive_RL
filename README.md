# Hive_RL

## Mission

**Hive_RL is an interface for hosting and building reinforcement-learning agents
that competitively play the board game [Hive](https://en.wikipedia.org/wiki/Hive_(game)).**

The goals are:

1. **A programmable game engine** — accurate Hive rules with a clean Python API so RL agents can interact with the game state.
2. **A bot pool** — a local registry of bots that anyone can add to, identified by a name and a Python class.
3. **ELO tracking** — every match updates each bot's rating so relative strength is recorded over time.
4. **A tournament runner** — automate round-robin competitions between all registered bots.
5. **A simple CLI** — `hivesim register / list / match / tournament` to manage the pool and kick off games without writing any glue code.

The intended workflow is:

```
write a bot  →  register it locally  →  play it against other bots  →  track ELO
```

---

## Implementation Plan

## Decision: build in [HiveSim](https://github.com/Devans90/HiveSim)

After reviewing available options (hive-hydra is not published to PyPI; no other
maintained Hive libraries exist with a programmable Python interface), **HiveSim
is the right and only practical foundation** for this project.

HiveSim already provides:

| Component | Status |
|---|---|
| Complete Hive game engine (all 5 standard pieces, stacking, one-hive rule) | ✅ done |
| `BaseBot` abstract class + `RandomBot` reference implementation | ✅ done |
| `simulate_game(white_bot, black_bot, …) → (winner, turns, game)` | ✅ done |
| Gymnasium-compatible `HiveEnv` for RL training | ✅ done |
| Plotly visualisation | ✅ done |
| Game logging (`GameLogger`) | ✅ done |
| Full test suite | ✅ done |

What is **missing** — and what this plan specifies — is the **competitive pool layer**:

| Component | Status |
|---|---|
| ELO rating calculations | ❌ to add |
| Persistent bot registry (local JSON pool) | ❌ to add |
| Match runner with automatic ELO updates | ❌ to add |
| Round-robin tournament runner | ❌ to add |
| CLI: `register`, `unregister`, `list`, `match`, `tournament` | ❌ to add |

Because all new code depends entirely on HiveSim and adds no value as a
separate package, **all of the following changes should be made directly in the
HiveSim repository**.  This `Hive_RL` repo serves only as a tracking document
for that plan.

---

## Plan for HiveSim

### 1  New files

```
src/hivesim/
    elo.py          # pure ELO maths (no I/O)
    pool.py         # BotEntry dataclass + BotPool (JSON persistence)
    tournament.py   # run_match / run_tournament (wraps simulate_game)
    cli.py          # argparse CLI entry-point
tests/
    test_elo.py
    test_pool.py
    test_tournament.py
```

### 2  Modified files

| File | Change |
|---|---|
| `pyproject.toml` | add CLI script entry-point |
| `src/hivesim/__init__.py` | export new public symbols |

---

## Detailed specifications

### `src/hivesim/elo.py`

Pure functions, no side-effects.

```python
DEFAULT_K:      float = 32.0
DEFAULT_RATING: float = 1200.0

def expected_score(rating_a: float, rating_b: float) -> float:
    """P(A beats B) = 1 / (1 + 10^((Rb-Ra)/400))"""

def update_ratings(
    rating_a: float,
    rating_b: float,
    winner: str | None,       # "white" | "black" | None (draw)
    k: float = DEFAULT_K,
) -> tuple[float, float]:
    """Return (new_rating_a, new_rating_b)."""
```

### `src/hivesim/pool.py`

```python
@dataclass
class BotEntry:
    name:         str
    module:       str           # importable Python path, e.g. "my_bots.greedy"
    class_name:   str           # class inside that module
    elo:          float = 1200.0
    games_played: int   = 0
    wins:         int   = 0
    losses:       int   = 0
    draws:        int   = 0

class BotPool:
    """JSON-backed registry of bots and their ELO ratings.

    Default pool file: pool.json  (current working directory).
    Bots are loaded dynamically; the only requirement is that the
    bot class supports BotClass(team=team, name=name) — exactly the
    same signature as BaseBot.__init__.
    """
    def __init__(self, pool_file: str | Path = "pool.json") -> None: ...

    # persistence
    def save(self) -> None: ...

    # CRUD
    def register(self, name: str, module: str, class_name: str,
                 elo: float = 1200.0) -> BotEntry: ...
    def unregister(self, name: str) -> None: ...          # KeyError if missing
    def get(self, name: str) -> BotEntry: ...             # KeyError if missing
    def list_bots(self) -> list[BotEntry]: ...            # sorted by ELO desc

    # dynamic instantiation
    def load_bot(self, name: str, team: str) -> BaseBot: ...

    # result recording (called by runner, not directly by users)
    def record_result(self, name: str, new_elo: float,
                      outcome: str) -> None: ...          # "win"|"loss"|"draw"
```

**Pool JSON format** (`pool.json`):

```json
{
  "RandomBot": {
    "name":         "RandomBot",
    "module":       "hivesim.robots",
    "class_name":   "RandomBot",
    "elo":          1214.5,
    "games_played": 10,
    "wins":         6,
    "losses":       3,
    "draws":        1
  }
}
```

### `src/hivesim/tournament.py`

```python
@dataclass
class MatchResult:
    white_name:       str
    black_name:       str
    winner:           str | None   # "white" | "black" | None
    turns:            int
    white_elo_before: float
    black_elo_before: float
    white_elo_after:  float
    black_elo_after:  float

@dataclass
class TournamentResult:
    match_results: list[MatchResult]

    @property
    def total_games(self) -> int: ...

def run_match(
    pool:       BotPool,
    white_name: str,
    black_name: str,
    games:      int  = 1,
    verbose:    bool = False,
) -> list[MatchResult]:
    """Play *games* games (white_name always white, black_name always black).
    ELO is updated in pool and pool.save() is called after every game.
    """

def run_tournament(
    pool:             BotPool,
    games_per_side:   int  = 1,   # each ordered pair plays this many games
    verbose:          bool = False,
) -> TournamentResult:
    """Round-robin: every ordered pair (A-white, B-black) and (B-white, A-black)
    each plays *games_per_side* games, so each unordered pair plays
    2 × games_per_side games in total.
    Requires ≥ 2 bots in pool.
    """
```

### `src/hivesim/cli.py`

Entry-point: `hivesim` (registered in `pyproject.toml`).

```
hivesim [--pool FILE] <command> [options]

Commands
--------
register    --name NAME --module MODULE --class CLASS [--elo ELO]
unregister  --name NAME
list                        # show leaderboard
match       --white BOT --black BOT [--games N] [--verbose]
tournament  [--games-per-side N] [--verbose]
```

**Leaderboard output** (`hivesim list`):

```
#    Name                  ELO      W     L     D    GP
----------------------------------------------------------
1    GreedyBot            1243    8     3     1    12
2    RandomBot            1187    4     7     1    12
```

---

## `pyproject.toml` changes

```toml
[project.scripts]
hivesim = "hivesim.cli:main"
```

---

## Example workflows

### Register and play

```bash
# install
pip install -e .

# register the built-in random bot as a baseline
hivesim register --name Random --module hivesim.robots --class RandomBot

# register your own bot (must be importable from cwd)
hivesim register --name MyBot --module my_bots.greedy --class GreedyBot

# play a 10-game match
hivesim match --white MyBot --black Random --games 10 --verbose

# view leaderboard
hivesim list
```

### Full round-robin tournament

```bash
hivesim tournament --games-per-side 2 --verbose
```

### Programmatic usage

```python
from hivesim.pool import BotPool
from hivesim.tournament import run_match, run_tournament

pool = BotPool("pool.json")
pool.register("Random", "hivesim.robots", "RandomBot")
pool.register("Greedy", "my_bots.greedy", "GreedyBot")

results = run_tournament(pool, games_per_side=2, verbose=True)
for bot in pool.list_bots():
    print(f"{bot.name}: {bot.elo:.0f} ({bot.wins}W {bot.losses}L {bot.draws}D)")
```

### Writing a bot

```python
# my_bots/greedy.py
import random
from hivesim.robots import BaseBot

class GreedyBot(BaseBot):
    def choose_action_type(self, can_move, can_place, game_state) -> str:
        return "move" if can_move else "place"

    def choose_piece_type(self, available_pieces, movable_pieces,
                          action_type, game_state) -> str:
        pool = movable_pieces if action_type == "move" else available_pieces
        return random.choice(list(pool.keys()))

    def choose_piece_id(self, piece_ids, piece_type, action_type, game_state) -> str:
        return random.choice(piece_ids)

    def choose_target_location(self, available_spaces, piece_type,
                               action_type, game_state):
        return random.choice(available_spaces)
```

```bash
# register and play
hivesim register --name Greedy --module my_bots.greedy --class GreedyBot
hivesim match --white Greedy --black Random --games 5
```

---

## Tests to add in HiveSim

### `tests/test_elo.py`
- `expected_score` is in (0, 1) and symmetric: `E(a,b) + E(b,a) == 1`
- Equal ratings → expected score of 0.5
- Winner gains points, loser loses same amount (zero-sum)
- Draw with equal ratings → no change

### `tests/test_pool.py`
- Register / duplicate-register raises `ValueError`
- Unregister / missing-unregister raises `KeyError`
- `save()` + reload from disk preserves all fields
- `list_bots()` is sorted by ELO descending
- `load_bot()` returns correct `BaseBot` subclass
- `load_bot()` with bad module raises `ImportError`

### `tests/test_tournament.py`
- `run_match` updates both bots' ELO and saves pool
- Winner's ELO increases, loser's decreases, zero-sum per game
- Draw with equal ratings leaves ELO unchanged
- `run_tournament` requires ≥ 2 bots
- Round-robin: N bots → N*(N-1) ordered pairs, each with `games_per_side` games
