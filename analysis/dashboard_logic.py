from pathlib import Path
import re
import tomllib

import pandas as pd
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
    "RB": 4,
    "CM": 5,
    "LW": 6,
    "RW": 7,
    "ST": 8,
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
        "ST": [(40, 88), (60, 80)],
    },
}


def _money_millions(value: float) -> str:
    return f"EUR {value / 1e6:,.1f}M"


def _money_thousands(value: float) -> str:
    return f"EUR {value / 1e3:,.1f}K"


def _normalize_nationality_token(value) -> str:
    token = str(value).strip().lower()
    return re.sub(r"\s+", " ", token)


def _extract_nationality_from_description(description_text) -> str | None:
    if pd.isna(description_text):
        return None

    text = str(description_text).strip()
    if not text:
        return None

    match_obj = re.search(
        r"\bis an? ([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .'-]*?) footballer\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match_obj:
        return None

    nationality = str(match_obj.group(1)).strip()
    return nationality if nationality else None


def _coerce_bool_or_none(value) -> bool | None:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(int(value))

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


@st.cache_data(show_spinner=False)
def _load_foreign_policy() -> tuple[set[str], bool]:
    domestic_default = {"brazilian", "brazil", "brasil"}
    unknown_default = True

    if not EXPERIMENT_CONFIG_PATH.exists():
        return domestic_default, unknown_default

    try:
        with EXPERIMENT_CONFIG_PATH.open("rb") as handle:
            cfg = tomllib.load(handle)
        constraints = cfg.get("constraints", {})

        raw_domestic = constraints.get("domestic_nationalities", ["Brazilian", "Brazil", "Brasil"])
        unknown_is_foreign = bool(constraints.get("unknown_nationality_is_foreign", True))

        if isinstance(raw_domestic, list):
            candidates = raw_domestic
        else:
            candidates = [raw_domestic]

        domestic = {
            _normalize_nationality_token(item)
            for item in candidates
            if _normalize_nationality_token(item)
        }
        if not domestic:
            domestic = domestic_default

        return domestic, unknown_is_foreign
    except Exception:
        return domestic_default, unknown_default


@st.cache_data(show_spinner=False)
def _load_player_nationality_lookup() -> pd.DataFrame:
    if not PROCESSED_PLAYER_DATA_PATH.exists():
        return pd.DataFrame(columns=["player_id", "nationality"])

    try:
        lookup_df = pd.read_csv(
            PROCESSED_PLAYER_DATA_PATH,
            usecols=lambda c: c in {"player_id", "description", "country_name"},
        )
    except Exception:
        return pd.DataFrame(columns=["player_id", "nationality"])

    if "player_id" not in lookup_df.columns:
        return pd.DataFrame(columns=["player_id", "nationality"])

    lookup_df["player_id"] = pd.to_numeric(lookup_df["player_id"], errors="coerce")
    lookup_df = lookup_df.dropna(subset=["player_id"]).copy()
    lookup_df["player_id"] = lookup_df["player_id"].astype(int)

    if "description" not in lookup_df.columns:
        lookup_df["description"] = ""
    if "country_name" not in lookup_df.columns:
        lookup_df["country_name"] = ""

    lookup_df["nationality_from_description"] = lookup_df["description"].apply(_extract_nationality_from_description)
    lookup_df["country_name"] = lookup_df["country_name"].fillna("").astype(str).str.strip()
    lookup_df["nationality"] = lookup_df["nationality_from_description"].fillna("")
    missing_nat = lookup_df["nationality"].astype(str).str.strip() == ""
    lookup_df.loc[missing_nat, "nationality"] = lookup_df.loc[missing_nat, "country_name"]

    lookup_df["nationality"] = lookup_df["nationality"].fillna("").astype(str).str.strip()
    lookup_df.loc[lookup_df["nationality"] == "", "nationality"] = pd.NA

    lookup_df = lookup_df[["player_id", "nationality"]].drop_duplicates(subset=["player_id"], keep="first")
    return lookup_df


def _attach_nationality_info(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    if df.empty or "player_id" not in df.columns:
        return df

    domestic_set, unknown_is_foreign = _load_foreign_policy()
    lookup = _load_player_nationality_lookup()

    if not lookup.empty:
        df = df.merge(
            lookup.rename(columns={"nationality": "nationality_lookup"}),
            on="player_id",
            how="left",
        )

        if "nationality" in df.columns:
            current_nat = df["nationality"].fillna("").astype(str).str.strip()
            fallback_nat = df["nationality_lookup"].fillna("").astype(str).str.strip()
            df["nationality"] = current_nat.where(current_nat != "", fallback_nat)
        else:
            df["nationality"] = df["nationality_lookup"]

        df = df.drop(columns=["nationality_lookup"], errors="ignore")
    elif "nationality" not in df.columns:
        df["nationality"] = pd.NA

    nat_norm = df["nationality"].fillna("").astype(str).map(_normalize_nationality_token)
    nat_known = nat_norm != ""
    computed_is_foreign = (~nat_norm.isin(domestic_set)).where(nat_known, other=unknown_is_foreign)

    if "is_foreign" in df.columns:
        parsed_existing = df["is_foreign"].apply(_coerce_bool_or_none)
        missing_mask = parsed_existing.isna()
        parsed_existing.loc[missing_mask] = computed_is_foreign.loc[missing_mask]
        df["is_foreign"] = parsed_existing.astype(bool)
    else:
        df["is_foreign"] = computed_is_foreign.astype(bool)

    df["nationality"] = df["nationality"].fillna("").astype(str).str.strip()
    df.loc[df["nationality"] == "", "nationality"] = "Unknown"
    return df


def _foreign_status_text(value) -> str:
    flag = _coerce_bool_or_none(value)
    if flag is None:
        return "Unknown"
    return "Yes" if flag else "No"


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

        picked = _pick_best(regular_pool, pos, need, selected_index, only_allowed=True)
        selected_index.extend(picked)
        remaining_need = need - len(picked)

        if remaining_need > 0:
            picked = _pick_best(reinforcement_pool, pos, remaining_need, selected_index, only_allowed=True)
            selected_index.extend(picked)
            remaining_need -= len(picked)

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

    decisions = _attach_nationality_info(decisions)

    return decisions, budget, formation, tree_meta, is_stochastic, selected_mode


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
    sibling_options = [
        f"Node {int(row['node_id'])} | p={float(row.get('branch_probability', 0.0)):.3f} | cum={float(row.get('cumulative_probability', 0.0)):.3f}"
        for _, row in siblings.iterrows()
    ]
    sibling_map = dict(zip(sibling_options, sibling_option_ids))
    return siblings, sibling_options, sibling_option_ids, sibling_map


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
