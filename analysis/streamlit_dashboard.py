from pathlib import Path
import tomllib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SQUAD_DECISIONS_PATH = OUTPUT_DIR / "squad_decisions.csv"
BUDGET_EVOLUTION_PATH = OUTPUT_DIR / "budget_evolution.csv"
FORMATION_DIAGNOSTICS_PATH = OUTPUT_DIR / "formation_diagnostics.csv"
SQUAD_DECISIONS_NODES_PATH = OUTPUT_DIR / "squad_decisions_nodes.csv"
BUDGET_EVOLUTION_NODES_PATH = OUTPUT_DIR / "budget_evolution_nodes.csv"
FORMATION_DIAGNOSTICS_NODES_PATH = OUTPUT_DIR / "formation_diagnostics_nodes.csv"
TREE_METADATA_PATH = OUTPUT_DIR / "tree_metadata.csv"
PLAYER_NODE_AUDIT_PATH = ROOT / "data" / "processed" / "player_node_audit.csv"
PROCESSED_PLAYER_DATA_PATH = ROOT / "data" / "processed" / "processed_player_data.csv"
EXPERIMENT_CONFIG_PATH = ROOT / "config" / "experiment.toml"
DET_RESULT_PATHS = [SQUAD_DECISIONS_PATH, BUDGET_EVOLUTION_PATH]
STOCH_RESULT_PATHS = [SQUAD_DECISIONS_NODES_PATH, BUDGET_EVOLUTION_NODES_PATH]

FIELD_WIDTH_RATIO = 0.65
BASE_FIELD_X_SCALE = 1.15
FIELD_X_SCALE = BASE_FIELD_X_SCALE * FIELD_WIDTH_RATIO
FIELD_X_MAX = 100 * FIELD_X_SCALE
BASE_RESERVE_PANEL_X0 = 121
BASE_RESERVE_PANEL_X1 = 177
RESERVE_PANEL_X0 = BASE_RESERVE_PANEL_X0 * FIELD_WIDTH_RATIO
RESERVE_PANEL_X1 = RESERVE_PANEL_X0 + ((BASE_RESERVE_PANEL_X1 - BASE_RESERVE_PANEL_X0) * FIELD_WIDTH_RATIO)
RESERVE_LEFT_TEXT_X = RESERVE_PANEL_X0 + 2
RESERVE_OVR_X = RESERVE_PANEL_X1 - 4
RESERVE_EVO_X = RESERVE_OVR_X - 2
PLOT_X_MAX = RESERVE_PANEL_X1 + 4
PITCH_FIG_HEIGHT = 760
MINI_TREE_HEIGHT_RATIO = 0.56

POSITION_PRIORITY = {
    "GK": 1,
    "CB": 2,
    "LB": 3,
    "RB": 3,
    "CM": 4,
    "LW": 5,
    "RW": 5,
    "ST": 6,
}

FORMATION_POSITION_COORDS = {
    "433": {
        "GK": [(50, 8)],
        "CB": [(36, 26), (64, 26)],
        "LB": [(12, 35)],
        "RB": [(88, 35)],
        "CM": [(30, 52), (50, 58), (70, 52)],
        "LW": [(15, 84)],
        "RW": [(85, 84)],
        "ST": [(50, 92), (40, 90), (60, 90)],
    },
    "442": {
        "GK": [(50, 8)],
        "CB": [(36, 26), (64, 26)],
        "LB": [(12, 35)],
        "RB": [(88, 35)],
        "CM": [(38, 54), (62, 54), (50, 58)],
        "LW": [(16, 64)],
        "RW": [(84, 64)],
        "ST": [(44, 88), (56, 80), (50, 92)],
    },
    "532": {
        "GK": [(50, 8)],
        "CB": [(28, 27), (72, 27), (50, 24)],
        "LB": [(12, 40)],
        "RB": [(88, 40)],
        "CM": [(32, 54), (68, 54), (50, 58)],
        "LW": [(16, 64)],
        "RW": [(84, 64)],
        "ST": [(44, 88), (56, 80), (50, 92)],
    }
}


def _money_millions(value: float) -> str:
    return f"EUR {value / 1e6:,.1f}M"


def _money_thousands(value: float) -> str:
    return f"EUR {value / 1e3:,.1f}K"


@st.cache_data(show_spinner=False)
def _load_salary_cap_multiplier() -> float:
    default_multiplier = 1.2
    if not EXPERIMENT_CONFIG_PATH.exists():
        return default_multiplier

    try:
        with EXPERIMENT_CONFIG_PATH.open("rb") as handle:
            cfg = tomllib.load(handle)
        constraints = cfg.get("constraints", {})
        return float(constraints.get("salary_cap_multiplier_initial", default_multiplier))
    except Exception:
        return default_multiplier


def _compute_node_payroll_eur(window_df: pd.DataFrame, squad_col: str = "in_squad") -> float:
    if window_df.empty or "wage" not in window_df.columns:
        return 0.0

    active_col = squad_col if squad_col in window_df.columns else "in_squad"
    if active_col not in window_df.columns:
        return 0.0

    active_flags = pd.to_numeric(window_df[active_col], errors="coerce").fillna(0).astype(int)
    wages = pd.to_numeric(window_df["wage"], errors="coerce").fillna(0.0)
    return float((wages * active_flags).sum())


def _estimate_salary_cap_eur(decisions: pd.DataFrame, is_stochastic: bool) -> float | None:
    if decisions.empty or "window" not in decisions.columns:
        return None

    baseline = decisions.copy()
    baseline["window"] = pd.to_numeric(baseline["window"], errors="coerce")
    baseline = baseline[baseline["window"] == 0].copy()
    if baseline.empty:
        return None

    if is_stochastic and "node_id" in baseline.columns:
        baseline["node_id"] = pd.to_numeric(baseline["node_id"], errors="coerce")
        baseline = baseline.dropna(subset=["node_id"])
        if baseline.empty:
            return None
        root_node_id = int(baseline["node_id"].min())
        baseline = baseline[baseline["node_id"] == root_node_id].copy()

    root_payroll_eur = _compute_node_payroll_eur(baseline, squad_col="in_squad")
    return root_payroll_eur * _load_salary_cap_multiplier()


def _files_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _latest_mtime(paths: list[Path]) -> float:
    return max(path.stat().st_mtime if path.exists() else 0.0 for path in paths)


def _infer_dashboard_mode(preferred: str = "auto") -> str:
    det_ready = _files_exist(DET_RESULT_PATHS)
    stoch_ready = _files_exist(STOCH_RESULT_PATHS)

    if preferred == "deterministic":
        if not det_ready:
            raise FileNotFoundError("Deterministic files are missing. Run deterministic optimization first.")
        return "deterministic"

    if preferred == "stochastic":
        if not stoch_ready:
            raise FileNotFoundError("Stochastic files are missing. Run stochastic optimization first.")
        return "stochastic"

    if det_ready and stoch_ready:
        return "deterministic" if _latest_mtime(DET_RESULT_PATHS) >= _latest_mtime(STOCH_RESULT_PATHS) else "stochastic"
    if det_ready:
        return "deterministic"
    if stoch_ready:
        return "stochastic"

    missing = [
        str(path.relative_to(ROOT))
        for path in [
            SQUAD_DECISIONS_PATH,
            BUDGET_EVOLUTION_PATH,
            FORMATION_DIAGNOSTICS_PATH,
            SQUAD_DECISIONS_NODES_PATH,
            BUDGET_EVOLUTION_NODES_PATH,
            FORMATION_DIAGNOSTICS_NODES_PATH,
        ]
        if not path.exists()
    ]
    raise FileNotFoundError(
        "Missing output files: " + ", ".join(missing) + ". Run the optimization pipeline first."
    )


def _safe_int(value) -> int:
    if pd.isna(value):
        return 0
    return int(value)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(show_spinner=False)
def _load_node_audit_lookup() -> pd.DataFrame:
    if not PLAYER_NODE_AUDIT_PATH.exists():
        return pd.DataFrame()

    audit_df = pd.read_csv(PLAYER_NODE_AUDIT_PATH)
    required_cols = {"node_id", "player_id"}
    if not required_cols.issubset(audit_df.columns):
        return pd.DataFrame()

    if "stage" in audit_df.columns and "window" not in audit_df.columns:
        audit_df = audit_df.rename(columns={"stage": "window"})

    if "window" not in audit_df.columns:
        return pd.DataFrame()

    for col in ["node_id", "window", "player_id", "starter_allowed", "chemistry_multiplier"]:
        if col in audit_df.columns:
            audit_df[col] = pd.to_numeric(audit_df[col], errors="coerce")

    audit_df = audit_df.dropna(subset=["node_id", "window", "player_id"]).copy()
    audit_df["node_id"] = audit_df["node_id"].astype(int)
    audit_df["window"] = audit_df["window"].astype(int)
    audit_df["player_id"] = audit_df["player_id"].astype(int)

    keep_cols = ["node_id", "window", "player_id"]
    for optional_col in ["starter_allowed", "chemistry_multiplier"]:
        if optional_col in audit_df.columns:
            keep_cols.append(optional_col)

    return audit_df[keep_cols].drop_duplicates(subset=["node_id", "window", "player_id"], keep="last")


def _evolution_text(curr_ovr: int, prev_ovr: int | None) -> str:
    if prev_ovr is None:
        return ""
    delta = curr_ovr - prev_ovr
    if delta > 0:
        return f"(+{delta})"
    if delta < 0:
        return f"({delta})"
    return ""


def _evolution_badge(curr_ovr: int, prev_ovr: int | None) -> str:
    if prev_ovr is None:
        return ""

    delta = curr_ovr - prev_ovr
    if delta > 0:
        return f"<b><span style='color:#0B6E1A'>(+{delta})</span></b>"
    if delta < 0:
        return f"<b><span style='color:#D42424'>({delta})</span></b>"
    return ""


def _display_name(name: str, max_len: int = 26) -> str:
    parts = str(name).split()
    if len(parts) >= 2:
        compact = f"{parts[0]} {parts[-1]}"
    else:
        compact = str(name)

    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1] + "..."


def _origin_text(row: pd.Series) -> str:
    origin_club = row.get("origin_club", "Unknown")
    origin_league = row.get("origin_league", "")

    if pd.isna(origin_club) or str(origin_club).strip() == "":
        origin_club = "Unknown"

    if pd.isna(origin_league) or str(origin_league).strip() == "":
        return f"Origin: {origin_club}"

    return f"Origin: {origin_club} ({origin_league})"


def _position_key(pos: str) -> tuple[int, str]:
    return (POSITION_PRIORITY.get(pos, 99), pos)


def _get_position_coords(formation_scheme: str | None) -> dict[str, list[tuple[float, float]]]:
    if formation_scheme is None:
        return FORMATION_POSITION_COORDS["433"]
    scheme_key = str(formation_scheme).strip()
    return FORMATION_POSITION_COORDS.get(scheme_key, FORMATION_POSITION_COORDS["433"])


def _sort_reserves_by_priority(reserves: pd.DataFrame) -> pd.DataFrame:
    if reserves.empty:
        return reserves

    df = reserves.copy()
    df["pos_priority"] = df["pos_group"].map(POSITION_PRIORITY).fillna(99).astype(int)
    df["injury_sort"] = pd.to_numeric(df.get("injured", 0), errors="coerce").fillna(0).astype(int)
    df = df.sort_values(by=["injury_sort", "pos_priority", "ovr"], ascending=[True, True, False])
    return df.drop(columns=["pos_priority", "injury_sort"], errors="ignore")


def _with_post_decision_squad(window_df: pd.DataFrame) -> pd.DataFrame:
    df = window_df.copy()
    for col in ["in_squad", "bought", "sold"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if "bought_in_parent" in df.columns:
        df["bought_in_parent"] = pd.to_numeric(df["bought_in_parent"], errors="coerce").fillna(0).astype(int)
    else:
        df["bought_in_parent"] = 0

    if "bought_from_root" in df.columns:
        df["bought_from_root"] = pd.to_numeric(df["bought_from_root"], errors="coerce").fillna(0).astype(int)
    else:
        df["bought_from_root"] = 0

    # Display-only roster after node decisions: in_squad + bought - sold.
    df["in_squad_display"] = (df["in_squad"] + df["bought"] - df["sold"]).clip(lower=0, upper=1).astype(int)
    df["is_new_reinforcement"] = ((df["bought"] == 1) & (df["sold"] == 0)).astype(int)
    df["is_root_inherited_reinforcement"] = ((df["bought_from_root"] == 1) & (df["is_new_reinforcement"] == 0)).astype(int)
    return df


def _with_display_best_xi(window_df: pd.DataFrame, form_window: pd.DataFrame) -> pd.DataFrame:
    """Build a display-only best XI by OVR, respecting positional requirements and availability."""
    df = window_df.copy()
    df["is_starter_display"] = 0

    squad_col = "in_squad_display" if "in_squad_display" in df.columns else "in_squad"
    squad = df[df[squad_col] == 1].copy()
    if squad.empty:
        return df

    if "starter_allowed" not in squad.columns:
        squad["starter_allowed"] = 1
    squad["starter_allowed"] = pd.to_numeric(squad["starter_allowed"], errors="coerce").fillna(1).astype(int)

    if "is_new_reinforcement" not in squad.columns:
        squad["is_new_reinforcement"] = 0
    squad["is_new_reinforcement"] = pd.to_numeric(squad["is_new_reinforcement"], errors="coerce").fillna(0).astype(int)

    regular_pool = squad[squad["is_new_reinforcement"] != 1].copy()
    reinforcement_pool = squad[squad["is_new_reinforcement"] == 1].copy()

    if regular_pool.empty and reinforcement_pool.empty:
        return df

    req_map: dict[str, int] = {}
    if not form_window.empty and {"pos_group", "required_count"}.issubset(form_window.columns):
        req_series = (
            form_window[["pos_group", "required_count"]]
            .dropna(subset=["pos_group", "required_count"])
            .groupby("pos_group", as_index=True)["required_count"]
            .max()
        )
        req_map = {
            str(pos): int(cnt)
            for pos, cnt in req_series.items()
            if int(cnt) > 0
        }

    if not req_map:
        req_series = squad.groupby("pos_group", as_index=True)["is_starter"].sum()
        req_map = {
            str(pos): int(cnt)
            for pos, cnt in req_series.items()
            if int(cnt) > 0
        }

    target_total = int(sum(req_map.values())) if req_map else min(11, len(squad))
    selected_index: list[int] = []

    def _pick_best(pool: pd.DataFrame, pos: str, need: int, selected: list[int], only_allowed: bool = True) -> list[int]:
        if need <= 0 or pool.empty:
            return []

        candidates = pool[pool["pos_group"] == pos].copy()
        if only_allowed:
            candidates = candidates[candidates["starter_allowed"] == 1].copy()
        if candidates.empty:
            return []

        candidates = candidates.loc[~candidates.index.isin(selected)]
        if candidates.empty:
            return []

        candidates = candidates.sort_values(by=["ovr"], ascending=False)
        return candidates.head(need).index.tolist()

    for pos in sorted(req_map.keys(), key=lambda p: POSITION_PRIORITY.get(p, 99)):
        need = int(req_map[pos])
        if need <= 0:
            continue

        # 1) Prefer regular players available to start.
        picked = _pick_best(regular_pool, pos, need, selected_index, only_allowed=True)
        selected_index.extend(picked)
        remaining_need = need - len(picked)

        # 2) If the position is still uncovered, allow new signings in that same position.
        if remaining_need > 0:
            picked = _pick_best(reinforcement_pool, pos, remaining_need, selected_index, only_allowed=True)
            selected_index.extend(picked)
            remaining_need -= len(picked)

        # 3) Last-resort display fallback (unavailable players) to keep XI complete when possible.
        if remaining_need > 0:
            picked = _pick_best(regular_pool, pos, remaining_need, selected_index, only_allowed=False)
            selected_index.extend(picked)
            remaining_need -= len(picked)

        if remaining_need > 0:
            picked = _pick_best(reinforcement_pool, pos, remaining_need, selected_index, only_allowed=False)
            selected_index.extend(picked)

    selected_index = list(dict.fromkeys(selected_index))

    if len(selected_index) < target_total:
        remaining = squad.loc[~squad.index.isin(selected_index)].copy()
        remaining["pos_priority"] = remaining["pos_group"].map(POSITION_PRIORITY).fillna(99).astype(int)

        preferred = remaining[remaining["starter_allowed"] == 1].sort_values(
            by=["ovr", "pos_priority"],
            ascending=[False, True],
        )
        fallback = remaining[remaining["starter_allowed"] != 1].sort_values(
            by=["ovr", "pos_priority"],
            ascending=[False, True],
        )

        preferred_regular = preferred[preferred["is_new_reinforcement"] != 1]
        preferred_new = preferred[preferred["is_new_reinforcement"] == 1]
        fallback_regular = fallback[fallback["is_new_reinforcement"] != 1]
        fallback_new = fallback[fallback["is_new_reinforcement"] == 1]

        needed = target_total - len(selected_index)
        extra = preferred_regular.head(needed)
        if len(extra) < needed:
            extra = pd.concat([extra, preferred_new.head(needed - len(extra))], axis=0)
        if len(extra) < needed:
            extra = pd.concat([extra, fallback_regular.head(needed - len(extra))], axis=0)
        if len(extra) < needed:
            extra = pd.concat([extra, fallback_new.head(needed - len(extra))], axis=0)

        selected_index.extend(extra.index.tolist())
        selected_index = list(dict.fromkeys(selected_index))

    df.loc[df.index.isin(selected_index), "is_starter_display"] = 1
    return df


def _allocate_coord(pos: str, usage_count: int, formation_scheme: str | None = None) -> tuple[float, float]:
    formation_coords = _get_position_coords(formation_scheme)
    coords = formation_coords.get(pos, [(50, 50)])
    if usage_count < len(coords):
        x, y = coords[usage_count]
        return (x * FIELD_X_SCALE, y)

    base_x, base_y = coords[-1]
    overflow = usage_count - len(coords) + 1
    direction = -1 if overflow % 2 else 1
    return ((base_x + direction * (6 * overflow)) * FIELD_X_SCALE, base_y - 3)


@st.cache_data(show_spinner=False)
def load_data(preferred_mode: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, str]:
    selected_mode = _infer_dashboard_mode(preferred_mode)
    tree_meta = pd.DataFrame()

    if selected_mode == "deterministic":
        decisions = pd.read_csv(SQUAD_DECISIONS_PATH)
        budget = pd.read_csv(BUDGET_EVOLUTION_PATH)
        formation = pd.read_csv(FORMATION_DIAGNOSTICS_PATH)
        is_stochastic = False
    else:
        decisions = pd.read_csv(SQUAD_DECISIONS_NODES_PATH)
        budget = pd.read_csv(BUDGET_EVOLUTION_NODES_PATH)
        formation = pd.read_csv(FORMATION_DIAGNOSTICS_NODES_PATH)
        if TREE_METADATA_PATH.exists():
            tree_meta = pd.read_csv(TREE_METADATA_PATH)

        if "stage" in decisions.columns and "window" not in decisions.columns:
            decisions = decisions.rename(columns={"stage": "window"})
        if "stage" in budget.columns and "window" not in budget.columns:
            budget = budget.rename(columns={"stage": "window"})
        if "stage" in formation.columns and "window" not in formation.columns:
            formation = formation.rename(columns={"stage": "window"})

        if "formation_scheme" not in decisions.columns and "tactical_scheme" in decisions.columns:
            decisions["formation_scheme"] = decisions["tactical_scheme"]
        if "formation_scheme" not in formation.columns and "tactical_scheme" in formation.columns:
            formation["formation_scheme"] = formation["tactical_scheme"]
        is_stochastic = True

    decisions["window"] = decisions["window"].astype(int)
    budget["window"] = budget["window"].astype(int)
    formation["window"] = formation["window"].astype(int)

    for col in ["in_squad", "is_starter", "bought", "sold", "ovr"]:
        decisions[col] = decisions[col].fillna(0).astype(int)

    if is_stochastic and {"node_id", "window", "player_id"}.issubset(decisions.columns):
        decisions = decisions.drop_duplicates(subset=["node_id", "window", "player_id"], keep="last")
    elif {"window", "player_id"}.issubset(decisions.columns):
        decisions = decisions.drop_duplicates(subset=["window", "player_id"], keep="last")

    decisions["name"] = decisions["name"].fillna("Unknown")
    decisions["pos_group"] = decisions["pos_group"].fillna("UNK")
    if "origin_club" not in decisions.columns:
        decisions["origin_club"] = "Unknown"
    decisions["origin_club"] = decisions["origin_club"].fillna("Unknown")
    if "origin_league" not in decisions.columns:
        decisions["origin_league"] = ""
    decisions["origin_league"] = decisions["origin_league"].fillna("")

    node_audit_df = _load_node_audit_lookup() if is_stochastic else pd.DataFrame()

    if "starter_allowed" in decisions.columns:
        decisions["starter_allowed"] = pd.to_numeric(decisions["starter_allowed"], errors="coerce").fillna(1).astype(int)
    elif is_stochastic and not node_audit_df.empty and "starter_allowed" in node_audit_df.columns:
        injury_df = node_audit_df[["node_id", "window", "player_id", "starter_allowed"]].copy()
        injury_df["starter_allowed"] = pd.to_numeric(injury_df["starter_allowed"], errors="coerce")
        injury_df = injury_df.dropna(subset=["starter_allowed"])
        injury_df["starter_allowed"] = injury_df["starter_allowed"].astype(int)
        decisions = decisions.merge(injury_df, on=["node_id", "window", "player_id"], how="left")
        decisions["starter_allowed"] = decisions["starter_allowed"].fillna(1).astype(int)
    else:
        decisions["starter_allowed"] = 1

    if is_stochastic and not node_audit_df.empty and "chemistry_multiplier" in node_audit_df.columns:
        chemistry_df = node_audit_df[["node_id", "window", "player_id", "chemistry_multiplier"]].copy()
        chemistry_df["chemistry_multiplier"] = pd.to_numeric(chemistry_df["chemistry_multiplier"], errors="coerce")

        if "chemistry_multiplier" in decisions.columns:
            decisions["chemistry_multiplier"] = pd.to_numeric(decisions["chemistry_multiplier"], errors="coerce")
            decisions = decisions.merge(
                chemistry_df.rename(columns={"chemistry_multiplier": "chemistry_multiplier_audit"}),
                on=["node_id", "window", "player_id"],
                how="left",
            )
            decisions["chemistry_multiplier"] = decisions["chemistry_multiplier"].fillna(decisions["chemistry_multiplier_audit"])
            decisions = decisions.drop(columns=["chemistry_multiplier_audit"], errors="ignore")
        else:
            decisions = decisions.merge(chemistry_df, on=["node_id", "window", "player_id"], how="left")

    if "chemistry_multiplier" in decisions.columns:
        decisions["chemistry_multiplier"] = pd.to_numeric(decisions["chemistry_multiplier"], errors="coerce")

    if "injured" in decisions.columns:
        decisions["injured"] = pd.to_numeric(decisions["injured"], errors="coerce").fillna(0).astype(int)
    else:
        decisions["injured"] = (decisions["starter_allowed"] == 0).astype(int)

    return decisions, budget, formation, tree_meta, is_stochastic, selected_mode


def _injury_badge(row: pd.Series) -> str:
    injured = int(row.get("injured", 0)) == 1
    return "<b><span style='color:#D42424'>✚</span></b>" if injured else ""


def _reinforcement_badge(row: pd.Series) -> str:
    new_signing = int(row.get("is_new_reinforcement", 0)) == 1
    return "<span title='Novo Reforço' style='color:#0E9F6E; font-size:1.2em;'> 🖋</span>" if new_signing else ""


def _root_inherited_badge(row: pd.Series) -> str:
    inherited_root = int(row.get("is_root_inherited_reinforcement", 0)) == 1
    return "<span title='Comprado na Raiz (Here-and-Now)' style='color:#1D6FD9; font-size:1.15em;'> ↳</span>" if inherited_root else ""

def enrich_with_evolution(decisions: pd.DataFrame, tree_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    df = decisions.copy()

    if "ovr_prev" in df.columns:
        df["ovr_prev"] = pd.to_numeric(df["ovr_prev"], errors="coerce")
        if "ovr_delta" in df.columns:
            df["ovr_delta"] = pd.to_numeric(df["ovr_delta"], errors="coerce").fillna(0).astype(int)
        elif "ovr" in df.columns:
            df["ovr_delta"] = (df["ovr"] - df["ovr_prev"].fillna(df["ovr"])).astype(int)
        return df

    is_stochastic = "node_id" in df.columns
    has_tree = tree_meta is not None and not tree_meta.empty and {"node_id", "parent_id"}.issubset(tree_meta.columns)

    if is_stochastic and has_tree:
        tree = tree_meta[["node_id", "parent_id"]].copy()
        tree = tree.rename(columns={"parent_id": "parent_node_id"})
        tree["node_id"] = pd.to_numeric(tree["node_id"], errors="coerce")
        tree["parent_node_id"] = pd.to_numeric(tree["parent_node_id"], errors="coerce")

        df["node_id"] = pd.to_numeric(df["node_id"], errors="coerce")
        df = df.merge(tree, on="node_id", how="left")

        prev = df[["player_id", "node_id", "ovr"]].copy()
        prev = prev.rename(columns={"node_id": "parent_node_id", "ovr": "ovr_prev"})
        df = df.merge(prev, on=["player_id", "parent_node_id"], how="left")
        df = df.drop(columns=["parent_node_id"])
    else:
        prev = df[["player_id", "window", "ovr"]].copy()
        prev["window"] = prev["window"] + 1
        prev = prev.rename(columns={"ovr": "ovr_prev"})
        df = df.merge(prev, on=["player_id", "window"], how="left")

    return df


def summarize_window_finance(window_df: pd.DataFrame, budget_df: pd.DataFrame, window: int, node_id: int | None = None) -> dict:
    row = budget_df[budget_df["window"] == window]
    if node_id is not None and "node_id" in budget_df.columns:
        row = row[row["node_id"] == node_id]
    cash = float(row.iloc[0]["cash_balance"]) if not row.empty else 0.0
    deficit = float(row.iloc[0]["deficit"]) if not row.empty else 0.0

    if not row.empty and {"transfer_spent", "transfer_sold", "buys_count", "sells_count"}.issubset(row.columns):
        return {
            "cash": cash,
            "deficit": deficit,
            "spent": float(row.iloc[0]["transfer_spent"]),
            "sold": float(row.iloc[0]["transfer_sold"]),
            "buys": int(row.iloc[0]["buys_count"]),
            "sells": int(row.iloc[0]["sells_count"]),
        }

    transfers = window_df[window_df["window"] == window]
    spent = float(transfers.loc[transfers["bought"] == 1, "acquisition_cost"].sum())
    sold = float(transfers.loc[transfers["sold"] == 1, "market_value"].sum())
    buys = int((transfers["bought"] == 1).sum())
    sells = int((transfers["sold"] == 1).sum())

    return {
        "cash": cash,
        "deficit": deficit,
        "spent": spent,
        "sold": sold,
        "buys": buys,
        "sells": sells,
    }


def build_pitch_figure(window_df: pd.DataFrame, selected_window: int, formation_scheme: str | None = None) -> go.Figure:
    squad_col = "in_squad_display" if "in_squad_display" in window_df.columns else "in_squad"
    squad = window_df[window_df[squad_col] == 1].copy()
    starter_col = "is_starter_display" if "is_starter_display" in squad.columns else "is_starter"
    starters = squad[squad[starter_col] == 1].copy()
    reserves = squad[squad[starter_col] == 0].copy()

    starters = starters.sort_values(by=["pos_group", "ovr"], ascending=[True, False])
    reserves = _sort_reserves_by_priority(reserves)

    active_coords = _get_position_coords(formation_scheme)
    usage = {k: 0 for k in active_coords}
    starter_x, starter_y = [], []
    starter_ovr, starter_hover, starter_label = [], [], []

    for _, row in starters.iterrows():
        pos = row["pos_group"]
        coord = _allocate_coord(pos, usage.get(pos, 0), formation_scheme)
        usage[pos] = usage.get(pos, 0) + 1

        starter_x.append(coord[0])
        starter_y.append(coord[1])

        ovr_now = _safe_int(row["ovr"])
        ovr_prev = None if pd.isna(row["ovr_prev"]) else _safe_int(row["ovr_prev"])
        evo_txt = _evolution_text(ovr_now, ovr_prev)
        evo_badge = _evolution_badge(ovr_now, ovr_prev)
        injury_badge = _injury_badge(row)
        reinforcement_badge = _reinforcement_badge(row)
        root_inherited_badge = _root_inherited_badge(row)
        injury_status = "Injured" if int(row.get("injured", 0)) == 1 else "Available"
        if int(row.get("is_new_reinforcement", 0)) == 1:
            status_txt = "New Signing (Current Node)"
        elif int(row.get("is_root_inherited_reinforcement", 0)) == 1:
            status_txt = "Inherited from Root Purchase"
        else:
            status_txt = injury_status
        name_short = _display_name(str(row["name"]), max_len=20)
        salary_txt = _money_thousands(_safe_float(row.get("wage", 0.0)))
        chemistry_val = row.get("chemistry_multiplier", pd.NA)
        chemistry_txt = "N/A" if pd.isna(chemistry_val) else f"{_safe_float(chemistry_val):.2f} (nó)"

        starter_ovr.append(str(ovr_now))
        starter_label.append(f"<b>{name_short}</b> {injury_badge} {reinforcement_badge} {root_inherited_badge} {evo_badge}".strip())
        starter_hover.append(
            "<br>".join(
                [
                    f"<b>{row['name']}</b>",
                    f"Position: {pos}",
                    f"Status: {status_txt}",
                    f"OVR: {ovr_now} {evo_txt}".strip(),
                    f"Chemistry: {chemistry_txt}",
                    f"Salary: {salary_txt}",
                    _origin_text(row),
                    f"Market Value: {_money_millions(float(row['market_value']))}",
                ]
            )
        )

    reserve_left_text, reserve_evo_text, reserve_ovr_text = [], [], []
    reserve_hover, reserve_left_x, reserve_evo_x, reserve_ovr_x, reserve_y = [], [], [], [], []
    start_y = 96
    step_y = 4.6 if len(reserves) > 20 else 5.2

    for idx, (_, row) in enumerate(reserves.iterrows()):
        ovr_now = _safe_int(row["ovr"])
        ovr_prev = None if pd.isna(row["ovr_prev"]) else _safe_int(row["ovr_prev"])
        evo_txt = _evolution_text(ovr_now, ovr_prev)
        evo_badge = _evolution_badge(ovr_now, ovr_prev)
        injury_badge = _injury_badge(row)
        reinforcement_badge = _reinforcement_badge(row)
        root_inherited_badge = _root_inherited_badge(row)
        injury_status = "Injured" if int(row.get("injured", 0)) == 1 else "Available"
        if int(row.get("is_new_reinforcement", 0)) == 1:
            status_txt = "New Signing (Current Node)"
        elif int(row.get("is_root_inherited_reinforcement", 0)) == 1:
            status_txt = "Inherited from Root Purchase"
        else:
            status_txt = injury_status
        reserve_name = _display_name(str(row["name"]), max_len=24)
        salary_txt = _money_thousands(_safe_float(row.get("wage", 0.0)))
        chemistry_val = row.get("chemistry_multiplier", pd.NA)
        chemistry_txt = "N/A" if pd.isna(chemistry_val) else f"{_safe_float(chemistry_val):.2f} (nó)"

        reserve_left_x.append(RESERVE_LEFT_TEXT_X)
        reserve_evo_x.append(RESERVE_EVO_X)
        reserve_ovr_x.append(RESERVE_OVR_X)
        reserve_y.append(start_y - idx * step_y)
        reserve_left_text.append(f"<b>{row['pos_group']}</b>  {reserve_name} {injury_badge} {reinforcement_badge} {root_inherited_badge}".strip())
        reserve_evo_text.append(evo_badge)
        reserve_ovr_text.append(f"<b>{ovr_now}</b>")
        reserve_hover.append(
            "<br>".join(
                [
                    f"<b>{row['name']}</b>",
                    f"Position: {row['pos_group']}",
                    f"Status: {status_txt}",
                    f"OVR: {ovr_now} {evo_txt}".strip(),
                    f"Chemistry: {chemistry_txt}",
                    f"Salary: {salary_txt}",
                    _origin_text(row),
                    f"Market Value: {_money_millions(float(row['market_value']))}",
                ]
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=starter_x,
            y=starter_y,
            mode="markers+text",
            text=starter_ovr,
            textposition="middle center",
            textfont=dict(size=12, color="white", family="Arial Black"),
            marker=dict(size=28, color="#C8102E", line=dict(width=2, color="black")),
            hovertext=starter_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=starter_x,
            y=[y - 6 for y in starter_y],
            mode="text",
            text=starter_label,
            textposition="middle center",
            textfont=dict(size=14, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=reserve_left_x,
            y=reserve_y,
            mode="text",
            text=reserve_left_text,
            textposition="middle right",
            textfont=dict(size=12, color="black"),
            hovertext=reserve_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=reserve_evo_x,
            y=reserve_y,
            mode="text",
            text=reserve_evo_text,
            textposition="middle left",
            textfont=dict(size=13, color="black"),
            hovertext=reserve_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=reserve_ovr_x,
            y=reserve_y,
            mode="text",
            text=reserve_ovr_text,
            textposition="middle right",
            textfont=dict(size=13, color="black"),
            hovertext=reserve_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=FIELD_X_MAX, y1=100, fillcolor="#4CAF50", line=dict(color="white", width=2), layer="below"),
        dict(type="line", x0=0, y0=50, x1=FIELD_X_MAX, y1=50, line=dict(color="white", width=2), layer="below"),
        dict(type="circle", x0=(40 * FIELD_X_SCALE), y0=40, x1=(60 * FIELD_X_SCALE), y1=60, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=(30 * FIELD_X_SCALE), y0=84, x1=(70 * FIELD_X_SCALE), y1=100, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=(40 * FIELD_X_SCALE), y0=94, x1=(60 * FIELD_X_SCALE), y1=100, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=(30 * FIELD_X_SCALE), y0=0, x1=(70 * FIELD_X_SCALE), y1=16, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=(40 * FIELD_X_SCALE), y0=0, x1=(60 * FIELD_X_SCALE), y1=6, line=dict(color="white", width=2), layer="below"),
        dict(type="rect", x0=RESERVE_PANEL_X0, y0=0, x1=RESERVE_PANEL_X1, y1=100, fillcolor="#ECECEC", line=dict(width=2, color="black"), layer="below"),
    ]

    fig.update_layout(
        title=f"Squad Field View - Window {selected_window}",
        width=1650,
        height=PITCH_FIG_HEIGHT,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="#2B2B2B",
        paper_bgcolor="#2B2B2B",
        font=dict(color="white"),
        xaxis=dict(
            range=[-2, PLOT_X_MAX],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
            fixedrange=True,
            constrain="domain",
        ),
        yaxis=dict(
            range=[-6, 104],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        shapes=shapes,
        annotations=[
            dict(x=(RESERVE_PANEL_X0 + 2), y=103, text="<b>RESERVES</b>", showarrow=False, font=dict(size=14, color="white"), xanchor="left"),
        ],
    )

    return fig


def build_window_tables(window_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    squad_col = "in_squad_display" if "in_squad_display" in window_df.columns else "in_squad"
    squad = window_df[window_df[squad_col] == 1].copy()
    starter_col = "is_starter_display" if "is_starter_display" in squad.columns else "is_starter"
    starters = squad[squad[starter_col] == 1].copy()
    reserves = squad[squad[starter_col] == 0].copy()

    for df in [starters, reserves]:
        if "ovr_prev" not in df.columns:
            df["ovr_prev"] = df["ovr"]
        df["ovr_prev"] = df["ovr_prev"].fillna(df["ovr"])
        if "ovr_delta" not in df.columns:
            df["ovr_delta"] = (df["ovr"] - df["ovr_prev"]).astype(int)
        else:
            df["ovr_delta"] = pd.to_numeric(df["ovr_delta"], errors="coerce").fillna(0).astype(int)

    starters = starters.sort_values(by=["pos_group", "ovr"], ascending=[True, False])
    reserves = _sort_reserves_by_priority(reserves)

    starter_cols = ["name", "pos_group", "ovr", "ovr_delta", "market_value", "acquisition_cost"]
    reserve_cols = ["name", "pos_group", "ovr", "ovr_delta", "market_value"]

    starters = starters[starter_cols].rename(columns={"name": "player", "pos_group": "pos"})
    reserves = reserves[reserve_cols].rename(columns={"name": "player", "pos_group": "pos"})

    starters["market_value"] = starters["market_value"].apply(_money_millions)
    starters["acquisition_cost"] = starters["acquisition_cost"].apply(_money_millions)
    reserves["market_value"] = reserves["market_value"].apply(_money_millions)

    return starters, reserves


def _prepare_tree_meta(tree_meta: pd.DataFrame) -> pd.DataFrame:
    if tree_meta is None or tree_meta.empty:
        return pd.DataFrame()

    required = {"node_id", "stage"}
    if not required.issubset(tree_meta.columns):
        return pd.DataFrame()

    df = tree_meta.copy()
    for col in ["node_id", "parent_id", "stage", "branch_probability", "cumulative_probability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["node_id", "stage"])
    df["node_id"] = df["node_id"].astype(int)
    df["stage"] = df["stage"].astype(int)
    if "parent_id" in df.columns:
        df["parent_id"] = df["parent_id"].astype("Int64")
    return df


def _build_node_label(row: pd.Series) -> str:
    node_id = int(row["node_id"])
    branch = float(row.get("branch_probability", 0.0))
    cumulative = float(row.get("cumulative_probability", 0.0))
    parent_val = row.get("parent_id")
    parent_txt = "ROOT" if pd.isna(parent_val) else str(int(parent_val))
    scheme = row.get("tactical_scheme", "N/A")
    path = str(row.get("path", ""))
    scenario_label = row.get("scenario_label", "")
    if pd.isna(scenario_label):
        scenario_label = ""
    scenario_label = str(scenario_label).strip()

    base = f"Node {node_id} | p={branch:.3f} | cum={cumulative:.3f} | parent={parent_txt} | scheme={scheme} | {path}"
    if scenario_label:
        return f"{scenario_label} | {base}"
    return base


def _centered_tree_y_coords(df: pd.DataFrame) -> dict[int, float]:
    node_path = {
        int(row["node_id"]): str(row.get("path", ""))
        for _, row in df.iterrows()
    }

    children_by_parent: dict[int, list[int]] = {}
    for _, row in df.iterrows():
        if pd.isna(row.get("parent_id")):
            continue
        parent_id = int(row["parent_id"])
        node_id = int(row["node_id"])
        children_by_parent.setdefault(parent_id, []).append(node_id)

    for parent_id, children in children_by_parent.items():
        children_by_parent[parent_id] = sorted(
            children,
            key=lambda nid: (
                node_path.get(nid, ""),
                nid,
            ),
        )

    all_nodes = sorted(df["node_id"].astype(int).tolist())
    leaf_nodes = [nid for nid in all_nodes if nid not in children_by_parent]
    leaf_nodes = sorted(leaf_nodes, key=lambda nid: (node_path.get(nid, ""), nid))

    y_coords: dict[int, float] = {}
    max_leaf_rank = max(1, len(leaf_nodes))
    for idx, node_id in enumerate(leaf_nodes):
        y_coords[node_id] = float(max_leaf_rank - idx)

    stage_order_desc = sorted(df["stage"].unique().tolist(), reverse=True)
    fallback_cursor = 0.0
    for stage in stage_order_desc:
        stage_nodes = sorted(df[df["stage"] == stage]["node_id"].astype(int).tolist())
        for node_id in stage_nodes:
            if node_id in y_coords:
                continue

            children = [child for child in children_by_parent.get(node_id, []) if child in y_coords]
            if children:
                y_coords[node_id] = sum(y_coords[child] for child in children) / len(children)
            else:
                y_coords[node_id] = fallback_cursor
                fallback_cursor -= 1.0

    return y_coords


def _get_sibling_node_ids(tree_meta: pd.DataFrame, selected_node: int) -> list[int]:
    if tree_meta.empty or "parent_id" not in tree_meta.columns:
        return []

    selected_rows = tree_meta[tree_meta["node_id"] == int(selected_node)]
    if selected_rows.empty:
        return []

    selected_row = selected_rows.iloc[0]
    parent_val = selected_row.get("parent_id")
    if pd.isna(parent_val):
        return []

    selected_stage = int(selected_row.get("stage", -1))
    parent_id = int(parent_val)

    siblings = tree_meta[
        (tree_meta["parent_id"] == parent_id)
        & (tree_meta["stage"] == selected_stage)
        & (tree_meta["node_id"] != int(selected_node))
    ]["node_id"].dropna().astype(int).tolist()

    return sorted(set(siblings))


def _build_sibling_choices(tree_meta: pd.DataFrame, sibling_ids: list[int]) -> tuple[pd.DataFrame, list[str], list[int], dict[str, int]]:
    if tree_meta.empty or not sibling_ids:
        return pd.DataFrame(), [], [], {}

    siblings = tree_meta[tree_meta["node_id"].isin(sibling_ids)].copy()
    if siblings.empty:
        return pd.DataFrame(), [], [], {}

    if "cumulative_probability" in siblings.columns:
        siblings = siblings.sort_values(by=["cumulative_probability", "node_id"], ascending=[False, True])
    else:
        siblings = siblings.sort_values(by=["node_id"], ascending=[True])

    sibling_option_ids = siblings["node_id"].astype(int).tolist()
    sibling_options = [_build_node_label(row) for _, row in siblings.iterrows()]
    sibling_map = dict(zip(sibling_options, sibling_option_ids))
    return siblings, sibling_options, sibling_option_ids, sibling_map


def _extract_clicked_node(chart_state, valid_nodes: set[int] | None = None) -> int | None:
    if chart_state is None:
        return None

    selection = chart_state.get("selection", {}) if isinstance(chart_state, dict) else {}
    points = selection.get("points", []) if isinstance(selection, dict) else []
    if not points:
        return None

    for point in reversed(points):
        curve_number = point.get("curve_number")
        if curve_number != 1:
            continue

        custom = point.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            custom = custom[0]
        if custom is None:
            continue

        try:
            node_id = int(custom)
        except (TypeError, ValueError):
            continue

        if valid_nodes is None or node_id in valid_nodes:
            return node_id

    return None


def build_scenario_tree_figure(
    tree_meta: pd.DataFrame,
    selected_node: int,
    selected_stage: int,
    mini_mode: bool = False,
    mini_height: int | None = None,
) -> go.Figure:
    df = _prepare_tree_meta(tree_meta)

    if df.empty or "parent_id" not in df.columns:
        return go.Figure()

    stage_order = sorted(df["stage"].unique().tolist())
    x_map = {stage: idx for idx, stage in enumerate(stage_order)}
    y_coords = _centered_tree_y_coords(df)
    node_stage = {int(row["node_id"]): int(row["stage"]) for _, row in df.iterrows()}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    if "parent_id" in df.columns:
        for _, row in df.iterrows():
            if pd.isna(row["parent_id"]):
                continue

            node_id = int(row["node_id"])
            parent_id = int(row["parent_id"])
            if parent_id not in y_coords:
                continue

            parent_stage = node_stage.get(parent_id, int(row["stage"]) - 1)
            child_stage = int(row["stage"])
            edge_x.extend([x_map[parent_stage], x_map[child_stage], None])
            edge_y.extend([y_coords[parent_id], y_coords[node_id], None])

    marker_base = 12 if mini_mode else 18
    marker_scale = 28 if mini_mode else 42
    node_x, node_y, marker_size, marker_color, hover_text, node_text, node_custom = [], [], [], [], [], [], []
    for _, row in df.iterrows():
        node_id = int(row["node_id"])
        stage = int(row["stage"])
        cumulative = float(row.get("cumulative_probability", 0.0))
        branch = float(row.get("branch_probability", 0.0))

        node_x.append(float(x_map[stage]))
        node_y.append(y_coords[node_id])
        marker_size.append(marker_base + marker_scale * max(0.0, cumulative))

        if node_id == selected_node:
            marker_color.append("#F39C12")
        elif stage == selected_stage:
            marker_color.append("#2AA198")
        else:
            marker_color.append("#5D6D7E")

        path = str(row.get("path", ""))
        tactical = row.get("tactical_scheme", "N/A")
        scenario_label = str(row.get("scenario_label", "")).strip()
        if scenario_label.lower() == "nan":
            scenario_label = ""
        hover_text.append(
            "<br>".join(
                [
                    f"<b>Node {node_id}</b>",
                    f"Label: {scenario_label}" if scenario_label else "Label: (none)",
                    f"Stage: {stage}",
                    f"Branch Probability: {branch:.3f}",
                    f"Cumulative Probability: {cumulative:.3f}",
                    f"Tactical scheme: {tactical}",
                    f"Path: {path}",
                ]
            )
        )
        node_text.append(f"N{node_id}")
        node_custom.append(node_id)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#95A5A6", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="middle center",
            customdata=node_custom,
            marker=dict(size=marker_size, color=marker_color, line=dict(width=1.5, color="#1C2833")),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    stage_ticks = [x_map[s] for s in stage_order]
    stage_labels = [f"Stage {s}" for s in stage_order]

    fig.update_layout(
        title=" Node Tree" if mini_mode else "Scenario Tree Map (Horizontal Flow)",
        width=None if mini_mode else max(950, 240 * max(1, len(stage_order))),
        height=mini_height if (mini_mode and mini_height is not None) else (max(240, 100 + 18 * max(1, len([nid for nid in y_coords if nid in df["node_id"].tolist()]))) if mini_mode else max(280, 120 + 32 * max(1, len([nid for nid in y_coords if nid in df["node_id"].tolist()])))),
        margin=dict(l=20, r=20, t=55, b=20),
        plot_bgcolor="#0F1720",
        paper_bgcolor="#0F1720",
        font=dict(color="white"),
        xaxis=dict(
            title="Decision Stage",
            tickmode="array",
            tickvals=stage_ticks,
            ticktext=stage_labels,
            showgrid=True,
            gridcolor="#1F2A37",
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, fixedrange=True),
        clickmode="event+select",
        dragmode="pan",
    )
    return fig


def build_sibling_diff_table(selected_df: pd.DataFrame, sibling_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = ["player_id", "name", "pos_group", "in_squad", "is_starter", "bought", "sold", "ovr", "market_value"]
    left = selected_df[keep_cols].copy()
    right = sibling_df[keep_cols].copy()

    left = left.rename(columns={
        "name": "name_sel",
        "pos_group": "pos_sel",
        "in_squad": "in_squad_sel",
        "is_starter": "is_starter_sel",
        "bought": "bought_sel",
        "sold": "sold_sel",
        "ovr": "ovr_sel",
        "market_value": "market_value_sel",
    })
    right = right.rename(columns={
        "name": "name_sib",
        "pos_group": "pos_sib",
        "in_squad": "in_squad_sib",
        "is_starter": "is_starter_sib",
        "bought": "bought_sib",
        "sold": "sold_sib",
        "ovr": "ovr_sib",
        "market_value": "market_value_sib",
    })

    merged = left.merge(right, on="player_id", how="outer")
    for col in [
        "in_squad_sel",
        "is_starter_sel",
        "bought_sel",
        "sold_sel",
        "in_squad_sib",
        "is_starter_sib",
        "bought_sib",
        "sold_sib",
    ]:
        merged[col] = merged[col].fillna(0).astype(int)

    merged["player"] = merged["name_sel"].fillna(merged["name_sib"]).fillna("Unknown")
    merged["pos"] = merged["pos_sel"].fillna(merged["pos_sib"]).fillna("UNK")
    merged["ovr"] = merged["ovr_sel"].fillna(merged["ovr_sib"]).fillna(0).astype(int)
    merged["market_value"] = merged["market_value_sel"].fillna(merged["market_value_sib"]).fillna(0.0)

    diff_mask = (
        (merged["in_squad_sel"] != merged["in_squad_sib"])
        | (merged["bought_sel"] != merged["bought_sib"])
        | (merged["sold_sel"] != merged["sold_sib"])
    )
    if "is_starter_sel" in merged.columns and "is_starter_sib" in merged.columns:
        diff_mask = diff_mask | (merged["is_starter_sel"] != merged["is_starter_sib"])

    diff = merged[diff_mask].copy()
    if diff.empty:
        return pd.DataFrame(columns=["branch_action", "player", "pos", "selected_branch", "sibling_branch", "ovr", "market_value"])

    diff["branch_action"] = ""
    red_mask = (
        ((diff["in_squad_sel"] == 0) & (diff["in_squad_sib"] == 1))
        | ((diff["sold_sel"] == 1) & (diff["sold_sib"] == 0))
    )
    green_mask = ~red_mask

    diff.loc[green_mask, "branch_action"] = "GREEN - Bought/kept in selected branch"
    diff.loc[red_mask, "branch_action"] = "RED - Sold/dispensed in selected branch"

    def _status_text(prefix: str, row: pd.Series) -> str:
        in_squad = int(row.get(f"in_squad_{prefix}", 0))
        starter = int(row.get(f"is_starter_{prefix}", 0))
        bought = int(row.get(f"bought_{prefix}", 0))
        sold = int(row.get(f"sold_{prefix}", 0))
        state = "IN" if in_squad == 1 else "OUT"
        starter_txt = "XI" if starter == 1 else "Bench"
        return f"{state} | {starter_txt} | buy={bought} | sell={sold}"

    diff["selected_branch"] = diff.apply(lambda row: _status_text("sel", row), axis=1)
    diff["sibling_branch"] = diff.apply(lambda row: _status_text("sib", row), axis=1)

    diff = diff.sort_values(by=["market_value", "ovr"], ascending=[False, False])
    diff["market_value"] = diff["market_value"].apply(_money_millions)
    return diff[["branch_action", "player", "pos", "selected_branch", "sibling_branch", "ovr", "market_value"]]


def _style_sibling_diff(df: pd.DataFrame):
    if df.empty:
        return df

    def style_row(row: pd.Series) -> list[str]:
        action = str(row["branch_action"])
        if action.startswith("GREEN"):
            return ["background-color: #E8F8EE; color: #0B6E1A; font-weight: 600;"] * len(row)
        if action.startswith("RED"):
            return ["background-color: #FDECEC; color: #A61E1E; font-weight: 600;"] * len(row)
        return [""] * len(row)

    return df.style.apply(style_row, axis=1)


def _starter_slice(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df is None or window_df.empty:
        return pd.DataFrame()

    squad_col = "in_squad_display" if "in_squad_display" in window_df.columns else "in_squad"
    starter_col = "is_starter_display" if "is_starter_display" in window_df.columns else "is_starter"
    squad = window_df[window_df[squad_col] == 1].copy()
    return squad[squad[starter_col] == 1].copy()


@st.cache_data(show_spinner=False)
def _load_player_age_lookup() -> pd.DataFrame:
    if not PROCESSED_PLAYER_DATA_PATH.exists():
        return pd.DataFrame(columns=["player_id", "age"])

    age_df = pd.read_csv(PROCESSED_PLAYER_DATA_PATH, usecols=["player_id", "age"])
    age_df["player_id"] = pd.to_numeric(age_df["player_id"], errors="coerce")
    age_df["age"] = pd.to_numeric(age_df["age"], errors="coerce")
    age_df = age_df.dropna(subset=["player_id"]).copy()
    age_df["player_id"] = age_df["player_id"].astype(int)
    return age_df.drop_duplicates(subset=["player_id"], keep="last")


def _ensure_age_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "player_id" not in out.columns:
        out["age"] = pd.NA
        return out

    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce")
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    age_lookup = _load_player_age_lookup()
    if age_lookup.empty:
        if "age" in out.columns:
            out["age"] = pd.to_numeric(out["age"], errors="coerce")
        else:
            out["age"] = pd.NA
        return out

    if "age" not in out.columns:
        out = out.merge(age_lookup, on="player_id", how="left")
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        return out

    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    if out["age"].isna().any():
        out = out.merge(age_lookup, on="player_id", how="left", suffixes=("", "_lookup"))
        out["age"] = out["age"].fillna(out["age_lookup"])
        out = out.drop(columns=["age_lookup"], errors="ignore")

    return out


def _compute_squad_profile_metrics(window_df: pd.DataFrame) -> dict[str, float]:
    starters = _starter_slice(window_df)
    if starters.empty:
        return {
            "Média de OVR": 0.0,
            "Potencial de Revenda": 0.0,
            "Idade Média": float("nan"),
            "Entrosamento Total": 0.0,
            "Custo Salarial": 0.0,
        }

    starters = _ensure_age_column(starters)
    starters["ovr"] = pd.to_numeric(starters.get("ovr", 0), errors="coerce").fillna(0)
    starters["market_value"] = pd.to_numeric(starters.get("market_value", 0), errors="coerce").fillna(0.0)
    starters["wage"] = pd.to_numeric(starters.get("wage", 0), errors="coerce").fillna(0.0)
    starters["in_squad"] = pd.to_numeric(starters.get("in_squad", 0), errors="coerce").fillna(0).astype(int)
    starters["age"] = pd.to_numeric(starters.get("age", pd.Series(dtype=float)), errors="coerce")

    age_mean = float(starters["age"].mean()) if starters["age"].notna().any() else float("nan")
    chemistry_total = float(starters["in_squad"].sum())

    return {
        "Média de OVR": float(starters["ovr"].mean()),
        "Potencial de Revenda": float(starters["market_value"].sum()),
        "Idade Média": age_mean,
        "Entrosamento Total": chemistry_total,
        "Custo Salarial": float(starters["wage"].sum()),
    }


def _format_profile_metric(metric: str, value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if metric in {"Média de OVR", "Idade Média"}:
        return f"{value:.1f}"
    if metric == "Potencial de Revenda":
        return _money_millions(float(value))
    if metric == "Custo Salarial":
        return _money_thousands(float(value))
    if metric == "Entrosamento Total":
        return f"{int(round(value))}"
    return f"{value:.2f}"


def _compute_profile_scale_bounds(
    decisions: pd.DataFrame,
    formation: pd.DataFrame,
    window: int,
    node_ids: list[int],
) -> dict[str, tuple[float, float]]:
    metrics_bucket: dict[str, list[float]] = {
        "Média de OVR": [],
        "Potencial de Revenda": [],
        "Idade Média": [],
        "Entrosamento Total": [],
        "Custo Salarial": [],
    }

    for node_id in sorted(set(int(n) for n in node_ids)):
        node_df = decisions[
            (decisions["window"] == window)
            & (decisions["node_id"] == node_id)
        ].copy()
        if node_df.empty:
            continue

        node_form = formation[formation["window"] == window].copy()
        if "node_id" in node_form.columns:
            node_form = node_form[node_form["node_id"] == node_id].copy()

        node_df = _with_post_decision_squad(node_df)
        node_df = _with_display_best_xi(node_df, node_form)
        node_metrics = _compute_squad_profile_metrics(node_df)

        for metric, value in node_metrics.items():
            if not pd.isna(value):
                metrics_bucket[metric].append(float(value))

    bounds: dict[str, tuple[float, float]] = {}
    for metric, values in metrics_bucket.items():
        if not values:
            continue
        bounds[metric] = (min(values), max(values))

    return bounds


def build_squad_profile_radar(
    selected_df: pd.DataFrame,
    sibling_df: pd.DataFrame,
    selected_node: int,
    sibling_node: int,
    chart_height: int = 300,
    scale_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[go.Figure, pd.DataFrame]:
    categories = [
        "Média de OVR",
        "Potencial de Revenda",
        "Idade Média",
        "Entrosamento Total",
        "Custo Salarial",
    ]

    selected_metrics = _compute_squad_profile_metrics(selected_df)
    sibling_metrics = _compute_squad_profile_metrics(sibling_df)

    selected_raw = [selected_metrics[k] for k in categories]
    sibling_raw = [sibling_metrics[k] for k in categories]

    min_visible_radius = 12.0
    selected_norm: list[float] = []
    sibling_norm: list[float] = []
    for metric, sel_v, sib_v in zip(categories, selected_raw, sibling_raw):
        metric_bounds = scale_bounds.get(metric) if scale_bounds else None
        if metric_bounds is not None:
            lo, hi = float(metric_bounds[0]), float(metric_bounds[1])
        else:
            lo = min(float(sel_v), float(sib_v)) if not (pd.isna(sel_v) or pd.isna(sib_v)) else 0.0
            hi = max(float(sel_v), float(sib_v)) if not (pd.isna(sel_v) or pd.isna(sib_v)) else 0.0

        if pd.isna(sel_v) and pd.isna(sib_v):
            selected_norm.append(0.0)
            sibling_norm.append(0.0)
            continue
        if pd.isna(sel_v):
            selected_norm.append(0.0)
            sibling_norm.append(100.0)
            continue
        if pd.isna(sib_v):
            selected_norm.append(100.0)
            sibling_norm.append(0.0)
            continue

        if hi == lo:
            selected_norm.append(50.0)
            sibling_norm.append(50.0)
            continue

        sel_scaled = 100.0 * (float(sel_v) - lo) / (hi - lo)
        sib_scaled = 100.0 * (float(sib_v) - lo) / (hi - lo)
        sel_scaled = max(0.0, min(100.0, sel_scaled))
        sib_scaled = max(0.0, min(100.0, sib_scaled))

        # Keep the lower profile visible as an actual polygon, not a center-collapsed line.
        selected_norm.append(min_visible_radius + ((100.0 - min_visible_radius) * sel_scaled / 100.0))
        sibling_norm.append(min_visible_radius + ((100.0 - min_visible_radius) * sib_scaled / 100.0))

    theta = categories + [categories[0]]
    sel_r = selected_norm + [selected_norm[0]]
    sib_r = sibling_norm + [sibling_norm[0]]
    sel_raw_fmt = [_format_profile_metric(m, v) for m, v in zip(categories, selected_raw)]
    sib_raw_fmt = [_format_profile_metric(m, v) for m, v in zip(categories, sibling_raw)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=sel_r,
            theta=theta,
            fill="toself",
            name=f"Nó {selected_node}",
            line=dict(color="#F39C12", width=2),
            fillcolor="rgba(243, 156, 18, 0.25)",
            customdata=sel_raw_fmt + [sel_raw_fmt[0]],
            hovertemplate=f"%{{theta}}<br>Nó {selected_node}: %{{customdata}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=sib_r,
            theta=theta,
            fill="toself",
            name=f"Nó {sibling_node}",
            line=dict(color="#2AA198", width=2),
            fillcolor="rgba(42, 161, 152, 0.25)",
            customdata=sib_raw_fmt + [sib_raw_fmt[0]],
            hovertemplate=f"%{{theta}}<br>Nó {sibling_node}: %{{customdata}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Perfil do Elenco Titular: Nó {selected_node} vs Nó {sibling_node}",
        height=max(260, chart_height),
        margin=dict(l=12, r=12, t=50, b=12),
        paper_bgcolor="#111821",
        plot_bgcolor="#111821",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0),
        polar=dict(
            bgcolor="#111821",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, ticks="", gridcolor="#3D4A5A"),
            angularaxis=dict(gridcolor="#2E3948"),
        ),
    )

    metrics_table = pd.DataFrame(
        {
            "Métrica": categories,
            f"Nó {selected_node}": sel_raw_fmt,
            f"Nó {sibling_node}": sib_raw_fmt,
        }
    )
    return fig, metrics_table


def main() -> None:
    st.set_page_config(page_title="Squad Field Dashboard", page_icon="⚽", layout="wide")

    mode_label = st.sidebar.selectbox(
        "Result Mode",
        options=["Auto", "Deterministic", "Stochastic"],
        index=0,
    )
    preferred_mode = mode_label.lower()

    try:
        decisions, budget, formation, tree_meta, is_stochastic, selected_mode = load_data(preferred_mode)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run: julia --project=. main.jl")
        return
    
    st.title("Squad Interactive Field")
    st.caption(
        "Interactive pitch view with starters, reserves, finance balance, and OVR evolution per window."
    )

    st.sidebar.caption(f"Active dataset: {selected_mode}")

    decisions = enrich_with_evolution(decisions, tree_meta)
    prepared_tree_meta = _prepare_tree_meta(tree_meta) if is_stochastic else pd.DataFrame()

    windows = sorted(decisions["window"].unique().tolist())
    time_label = "Stage" if is_stochastic else "Window"
    window_key = "selected_window"
    pending_window_key = "pending_selected_window"
    pending_node_key = "pending_selected_node"

    if pending_window_key in st.session_state:
        try:
            pending_window = int(st.session_state[pending_window_key])
        except (TypeError, ValueError):
            pending_window = None

        if pending_window is not None and pending_window in windows:
            st.session_state[window_key] = pending_window
        st.session_state.pop(pending_window_key, None)

    if pending_node_key in st.session_state and not prepared_tree_meta.empty:
        try:
            pending_node = int(st.session_state[pending_node_key])
        except (TypeError, ValueError):
            pending_node = None

        if pending_node is not None:
            pending_meta = prepared_tree_meta[prepared_tree_meta["node_id"] == pending_node]
            if not pending_meta.empty:
                pending_stage = int(pending_meta.iloc[0]["stage"])
                st.session_state[f"selected_node_stage_{pending_stage}"] = pending_node
        st.session_state.pop(pending_node_key, None)

    if window_key not in st.session_state or int(st.session_state[window_key]) not in windows:
        st.session_state[window_key] = min(windows)

    selected_window = int(
        st.sidebar.slider(
            time_label,
            min_value=min(windows),
            max_value=max(windows),
            step=1,
            key=window_key,
        )
    )

    selected_node = None
    node_row = None
    stage_meta = pd.DataFrame()
    if is_stochastic and "node_id" in decisions.columns:
        nodes_in_window = sorted(decisions.loc[decisions["window"] == selected_window, "node_id"].dropna().unique().tolist())
        if not nodes_in_window:
            st.warning("No stochastic nodes found for selected stage.")
            return

        if not prepared_tree_meta.empty:
            stage_meta = prepared_tree_meta[
                (prepared_tree_meta["stage"] == selected_window) & (prepared_tree_meta["node_id"].isin(nodes_in_window))
            ].copy()

        if not stage_meta.empty:
            if "cumulative_probability" in stage_meta.columns:
                stage_meta = stage_meta.sort_values(by="cumulative_probability", ascending=False)
            else:
                stage_meta = stage_meta.sort_values(by="node_id")

            node_labels = [_build_node_label(row) for _, row in stage_meta.iterrows()]
            node_ids_order = stage_meta["node_id"].astype(int).tolist()
            node_map = dict(zip(node_labels, node_ids_order))

            state_key = f"selected_node_stage_{selected_window}"
            if state_key not in st.session_state or int(st.session_state[state_key]) not in node_ids_order:
                st.session_state[state_key] = node_ids_order[0]

            active_node_id = int(st.session_state[state_key])
            selected_index = max(0, node_ids_order.index(active_node_id))
            selected_label = st.sidebar.selectbox("Scenario Node", options=node_labels, index=selected_index)
            selected_node = int(node_map[selected_label])
            st.session_state[state_key] = selected_node
            node_row = stage_meta[stage_meta["node_id"] == selected_node].iloc[0]
        else:
            selected_node = int(st.sidebar.selectbox("Scenario Node", options=nodes_in_window, index=0))

    window_df = decisions[decisions["window"] == selected_window].copy()
    if selected_node is not None:
        window_df = window_df[window_df["node_id"] == selected_node].copy()

    if window_df.empty:
        st.warning("No data found for the selected window.")
        return

    form_window = formation[formation["window"] == selected_window].copy()
    if selected_node is not None and "node_id" in form_window.columns:
        form_window = form_window[form_window["node_id"] == selected_node].copy()

    # Display the roster after node decisions, and then pick a display-only strongest XI.
    window_df = _with_post_decision_squad(window_df)
    window_df = _with_display_best_xi(window_df, form_window)

    scheme = window_df["formation_scheme"].dropna().iloc[0] if "formation_scheme" in window_df and not window_df["formation_scheme"].dropna().empty else "N/A"
    finance = summarize_window_finance(window_df, budget, selected_window, selected_node)
    sold_mask = pd.to_numeric(window_df.get("sold", 0), errors="coerce").fillna(0).astype(int) == 1
    sold_players_df = window_df[sold_mask].copy()
    current_payroll_eur = _compute_node_payroll_eur(window_df, squad_col="in_squad")
    salary_cap_eur = _estimate_salary_cap_eur(decisions, is_stochastic)
    selected_sibling_node: int | None = None

    show_tree_sidebar = (
        is_stochastic
        and selected_node is not None
        and not prepared_tree_meta.empty
        and {"node_id", "parent_id", "stage"}.issubset(prepared_tree_meta.columns)
    )

    if show_tree_sidebar:
        main_col, side_col = st.columns([3.0, 2.2], gap="large")

        with main_col:
            pitch_fig = build_pitch_figure(window_df, selected_window, scheme)
            st.plotly_chart(pitch_fig, width="stretch")

        with side_col:
            mini_tree_height = int(PITCH_FIG_HEIGHT * MINI_TREE_HEIGHT_RATIO)
            tree_fig = build_scenario_tree_figure(
                prepared_tree_meta,
                selected_node,
                selected_window,
                mini_mode=True,
                mini_height=mini_tree_height,
            )
            tree_key = "scenario_tree_minimap"

            try:
                st.plotly_chart(
                    tree_fig,
                    width="stretch",
                    key=tree_key,
                    on_select="rerun",
                    selection_mode="points",
                )
                clicked_node = _extract_clicked_node(st.session_state.get(tree_key))
                if clicked_node is not None and clicked_node != selected_node:
                    clicked_meta = prepared_tree_meta[prepared_tree_meta["node_id"] == int(clicked_node)]
                    if not clicked_meta.empty:
                        clicked_stage = int(clicked_meta.iloc[0]["stage"])
                        if clicked_stage in windows:
                            st.session_state[pending_window_key] = clicked_stage
                        st.session_state[pending_node_key] = int(clicked_node)
                    else:
                        st.session_state[pending_node_key] = int(clicked_node)
                    st.rerun()
            except TypeError:
                st.plotly_chart(tree_fig, width="stretch")

            col_saidas, col_financas = st.columns([1, 1], gap="medium")

            with col_saidas:
                st.subheader("Saídas do Elenco")
                if sold_players_df.empty:
                    st.info("Sem vendas neste nó.")
                else:
                    sold_table = sold_players_df[["name", "pos_group", "ovr", "market_value", "wage"]].copy()
                    sold_table = sold_table.rename(
                        columns={
                            "name": "Jogador",
                            "pos_group": "Pos",
                            "ovr": "OVR",
                            "market_value": "Valor de Venda",
                            "wage": "Salário",
                        }
                    )
                    sold_table["Valor de Venda"] = pd.to_numeric(sold_table["Valor de Venda"], errors="coerce").fillna(0.0).apply(_money_millions)
                    sold_table["Salário"] = pd.to_numeric(sold_table["Salário"], errors="coerce").fillna(0.0).apply(_money_thousands)
                    st.dataframe(sold_table, width="stretch", hide_index=True)

            with col_financas:
                st.subheader("Status Financeiro")
                st.metric("Orçamento de Transferência", _money_millions(finance["cash"]))
                if salary_cap_eur is None:
                    st.metric("Folha Salarial", _money_thousands(current_payroll_eur))
                else:
                    salary_slack_eur = salary_cap_eur - current_payroll_eur
                    st.metric(
                        "Folha Salarial",
                        _money_thousands(current_payroll_eur),
                        delta=f"{_money_thousands(salary_slack_eur)} vs teto",
                    )

        sibling_ids = _get_sibling_node_ids(prepared_tree_meta, selected_node)
        siblings = pd.DataFrame()
        sibling_options: list[str] = []
        sibling_option_ids: list[int] = []
        sibling_map: dict[str, int] = {}
        sibling_df = pd.DataFrame()

        if sibling_ids:
            siblings, sibling_options, sibling_option_ids, sibling_map = _build_sibling_choices(prepared_tree_meta, sibling_ids)

        col_radar, col_sibling = st.columns([1.5, 1], gap="large")

        with col_sibling:
            st.subheader("Sibling Comparison")
            if sibling_ids and sibling_options:
                sibling_state_key = f"selected_sibling_stage_{selected_window}_node_{selected_node}"
                if sibling_state_key not in st.session_state or int(st.session_state[sibling_state_key]) not in sibling_option_ids:
                    st.session_state[sibling_state_key] = sibling_option_ids[0]

                active_sibling = int(st.session_state[sibling_state_key])
                selected_sibling_index = max(0, sibling_option_ids.index(active_sibling))
                selected_sibling_label = st.selectbox(
                    "Nó Irmão",
                    options=sibling_options,
                    index=selected_sibling_index,
                    key=f"sibling_compare_box_{selected_window}_{selected_node}",
                )
                selected_sibling_node = int(sibling_map[selected_sibling_label])
                st.session_state[sibling_state_key] = selected_sibling_node

                sibling_df = decisions[
                    (decisions["window"] == selected_window)
                    & (decisions["node_id"] == selected_sibling_node)
                ].copy()
                sibling_form = formation[formation["window"] == selected_window].copy()
                if "node_id" in sibling_form.columns:
                    sibling_form = sibling_form[sibling_form["node_id"] == selected_sibling_node].copy()

                sibling_df = _with_post_decision_squad(sibling_df)
                sibling_df = _with_display_best_xi(sibling_df, sibling_form)

                diff_tbl = build_sibling_diff_table(window_df.copy(), sibling_df.copy())
                if diff_tbl.empty:
                    st.info("Sem diferenças de decisão entre os nós comparados.")
                else:
                    st.dataframe(_style_sibling_diff(diff_tbl), width="stretch", hide_index=True)
            else:
                st.info("Não há nó irmão disponível para comparação neste estágio.")

        with col_radar:
            st.subheader("Visualização de Radar")
            if selected_sibling_node is not None and not sibling_df.empty:
                scale_node_ids = (
                    stage_meta["node_id"].dropna().astype(int).tolist()
                    if not stage_meta.empty and "node_id" in stage_meta.columns
                    else [selected_node] + sibling_ids
                )
                scale_bounds = _compute_profile_scale_bounds(
                    decisions,
                    formation,
                    selected_window,
                    scale_node_ids,
                )

                radar_fig, radar_tbl = build_squad_profile_radar(
                    window_df,
                    sibling_df,
                    selected_node,
                    selected_sibling_node,
                    chart_height=320,
                    scale_bounds=scale_bounds,
                )
                st.plotly_chart(radar_fig, width="stretch")
                st.caption("Escala normalizada de 0 a 100 por métrica; tabela com valores absolutos.")
                st.dataframe(radar_tbl, width="stretch", hide_index=True)
            else:
                st.info("Selecione um nó irmão para exibir o radar comparativo.")
    else:
        pitch_fig = build_pitch_figure(window_df, selected_window, scheme)
        st.plotly_chart(pitch_fig, width="stretch")

    left, mid, right = st.columns([1, 1, 1])
    scheme_label = scheme if selected_node is None else f"{scheme} | Node {selected_node}"
    left.metric("Scheme", scheme_label)
    mid.metric("Cash", _money_millions(finance["cash"]))
    right.metric("Deficit", _money_millions(finance["deficit"]))

    if is_stochastic and selected_node is not None and not tree_meta.empty:
        node_meta = prepared_tree_meta[prepared_tree_meta["node_id"] == selected_node]
        if not node_meta.empty:
            node_row = node_meta.iloc[0] if node_row is None else node_row
            st.subheader("Scenario Context")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stage", str(int(node_row.get("stage", selected_window))))
            c2.metric("Node Probability", f"{float(node_row.get('branch_probability', 0.0)):.3f}")
            c3.metric("Cumulative Probability", f"{float(node_row.get('cumulative_probability', 0.0)):.3f}")
            parent_val = node_row.get("parent_id")
            parent_text = "ROOT" if pd.isna(parent_val) else str(int(parent_val))
            c4.metric("Parent Node", parent_text)

            children_count = int(node_row.get("children_count", 0))
            is_leaf = bool(node_row.get("is_leaf", False))
            tactical = node_row.get("tactical_scheme", "N/A")
            path = str(node_row.get("path", ""))
            scenario_label = str(node_row.get("scenario_label", "")).strip()
            st.caption(
                f"Node type: {'Leaf' if is_leaf else 'Internal'} | Children: {children_count} | Tactical scheme: {tactical} | Path: {path}"
            )
            if scenario_label and scenario_label.lower() != "nan":
                st.info(f"Scenario label: {scenario_label}")

            if {"stage", "node_id", "cumulative_probability"}.issubset(prepared_tree_meta.columns):
                stage_nodes = prepared_tree_meta[prepared_tree_meta["stage"] == node_row.get("stage")][["node_id", "cumulative_probability"]].copy()
                stage_nodes = stage_nodes.sort_values(by="cumulative_probability", ascending=False)
                st.dataframe(stage_nodes, width="stretch", hide_index=True)

    starters_tbl, reserves_tbl = build_window_tables(window_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Starting XI")
        st.dataframe(starters_tbl, width="stretch", hide_index=True)
    with col2:
        st.subheader("Reserves")
        st.dataframe(reserves_tbl, width="stretch", hide_index=True)

    st.subheader("Tactical Constraint Check")

    if not form_window.empty:
        form_window = form_window.sort_values(by="pos_group")
        st.dataframe(form_window[["formation_scheme", "pos_group", "required_count", "actual_starters", "slack_titular"]], width="stretch", hide_index=True)
    else:
        st.info("No tactical diagnostics found for this window.")

    st.markdown("---")
    st.code("streamlit run analysis/streamlit_dashboard.py", language="bash")


if __name__ == "__main__":
    main()
