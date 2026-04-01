from pathlib import Path

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
DET_RESULT_PATHS = [SQUAD_DECISIONS_PATH, BUDGET_EVOLUTION_PATH]
STOCH_RESULT_PATHS = [SQUAD_DECISIONS_NODES_PATH, BUDGET_EVOLUTION_NODES_PATH]

FIELD_X_SCALE = 1.15
FIELD_X_MAX = 100 * FIELD_X_SCALE
RESERVE_PANEL_X0 = 121
RESERVE_PANEL_X1 = 177
RESERVE_LEFT_TEXT_X = RESERVE_PANEL_X0 + 2
RESERVE_OVR_X = RESERVE_PANEL_X1 - 4
RESERVE_EVO_X = RESERVE_OVR_X - 2
FINANCE_PANEL_X0 = 181
FINANCE_PANEL_X1 = 215
PLOT_X_MAX = 220

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

POSITION_COORDS = {
    "GK": [(50, 8)],
    "CB": [(36, 26), (64, 26)],
    "LB": [(12, 35)],
    "RB": [(88, 35)],
    "CM": [(30, 52), (50, 57), (70, 52)],
    "LW": [(15, 84)],
    "RW": [(85, 84)],
    "ST": [(50, 92), (40, 90), (60, 90)],
}


def _money_millions(value: float) -> str:
    return f"EUR {value / 1e6:,.1f}M"


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


def _allocate_coord(pos: str, usage_count: int) -> tuple[float, float]:
    coords = POSITION_COORDS.get(pos, [(50, 50)])
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

    return decisions, budget, formation, tree_meta, is_stochastic, selected_mode


def enrich_with_evolution(decisions: pd.DataFrame, tree_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    df = decisions.copy()

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


def build_pitch_figure(window_df: pd.DataFrame, finance: dict, selected_window: int) -> go.Figure:
    squad = window_df[window_df["in_squad"] == 1].copy()
    starters = squad[squad["is_starter"] == 1].copy()
    reserves = squad[squad["is_starter"] == 0].copy()

    starters = starters.sort_values(by=["pos_group", "ovr"], ascending=[True, False])
    reserves = reserves.sort_values(by=["pos_group", "ovr"], ascending=[True, False])

    usage = {k: 0 for k in POSITION_COORDS}
    starter_x, starter_y = [], []
    starter_ovr, starter_hover, starter_label = [], [], []

    for _, row in starters.iterrows():
        pos = row["pos_group"]
        coord = _allocate_coord(pos, usage.get(pos, 0))
        usage[pos] = usage.get(pos, 0) + 1

        starter_x.append(coord[0])
        starter_y.append(coord[1])

        ovr_now = _safe_int(row["ovr"])
        ovr_prev = None if pd.isna(row["ovr_prev"]) else _safe_int(row["ovr_prev"])
        evo_txt = _evolution_text(ovr_now, ovr_prev)
        evo_badge = _evolution_badge(ovr_now, ovr_prev)
        name_short = _display_name(str(row["name"]), max_len=20)

        starter_ovr.append(str(ovr_now))
        starter_label.append(f"<b>{name_short}</b> {evo_badge}".strip())
        starter_hover.append(
            "<br>".join(
                [
                    f"<b>{row['name']}</b>",
                    f"Position: {pos}",
                    f"OVR: {ovr_now} {evo_txt}".strip(),
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
        reserve_name = _display_name(str(row["name"]), max_len=24)

        reserve_left_x.append(RESERVE_LEFT_TEXT_X)
        reserve_evo_x.append(RESERVE_EVO_X)
        reserve_ovr_x.append(RESERVE_OVR_X)
        reserve_y.append(start_y - idx * step_y)
        reserve_left_text.append(f"<b>{row['pos_group']}</b>  {reserve_name}".strip())
        reserve_evo_text.append(evo_badge)
        reserve_ovr_text.append(f"<b>{ovr_now}</b>")
        reserve_hover.append(
            "<br>".join(
                [
                    f"<b>{row['name']}</b>",
                    f"Position: {row['pos_group']}",
                    f"OVR: {ovr_now} {evo_txt}".strip(),
                    _origin_text(row),
                    f"Market Value: {_money_millions(float(row['market_value']))}",
                ]
            )
        )

    finance_header = "<b>FINANCE</b>"
    finance_body = "<br>".join(
        [
            f"Cash: {_money_millions(finance['cash'])}",
            f"Deficit: {_money_millions(finance['deficit'])}",
            f"Spent: {_money_millions(finance['spent'])}",
            f"Sold: {_money_millions(finance['sold'])}",
            f"Moves: +{finance['buys']} / -{finance['sells']}",
        ]
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

    fig.add_trace(
        go.Scatter(
            x=[198],
            y=[92],
            mode="text",
            text=[finance_header],
            textfont=dict(size=15, color="white"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[198],
            y=[81],
            mode="text",
            text=[finance_body],
            textfont=dict(size=12, color="#00FF88", family="Consolas"),
            hoverinfo="skip",
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
        dict(type="rect", x0=FINANCE_PANEL_X0, y0=65, x1=FINANCE_PANEL_X1, y1=100, fillcolor="#1F1F1F", line=dict(width=1, color="gray"), layer="below"),
    ]

    fig.update_layout(
        title=f"Squad Field View - Window {selected_window}",
        width=1650,
        height=760,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="#2B2B2B",
        paper_bgcolor="#2B2B2B",
        font=dict(color="white"),
        xaxis=dict(range=[-2, PLOT_X_MAX], showgrid=False, zeroline=False, showticklabels=False, visible=False, fixedrange=True),
        yaxis=dict(range=[-6, 104], showgrid=False, zeroline=False, showticklabels=False, visible=False, fixedrange=True),
        shapes=shapes,
        annotations=[
            dict(x=(RESERVE_PANEL_X0 + 2), y=103, text="<b>RESERVES</b>", showarrow=False, font=dict(size=14, color="white"), xanchor="left"),
        ],
    )

    return fig


def build_window_tables(window_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    squad = window_df[window_df["in_squad"] == 1].copy()
    starters = squad[squad["is_starter"] == 1].copy()
    reserves = squad[squad["is_starter"] == 0].copy()

    for df in [starters, reserves]:
        df["ovr_prev"] = df["ovr_prev"].fillna(df["ovr"])
        df["ovr_delta"] = (df["ovr"] - df["ovr_prev"]).astype(int)

    starters = starters.sort_values(by=["pos_group", "ovr"], ascending=[True, False])
    reserves = reserves.sort_values(by=["pos_group", "ovr"], ascending=[True, False])

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
    return f"Node {node_id} | p={branch:.3f} | cum={cumulative:.3f} | parent={parent_txt} | scheme={scheme} | {path}"


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


def build_scenario_tree_figure(tree_meta: pd.DataFrame, selected_node: int, selected_stage: int) -> go.Figure:
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

    node_x, node_y, marker_size, marker_color, hover_text, node_text, node_custom = [], [], [], [], [], [], []
    for _, row in df.iterrows():
        node_id = int(row["node_id"])
        stage = int(row["stage"])
        cumulative = float(row.get("cumulative_probability", 0.0))
        branch = float(row.get("branch_probability", 0.0))

        node_x.append(float(x_map[stage]))
        node_y.append(y_coords[node_id])
        marker_size.append(18 + 42 * max(0.0, cumulative))

        if node_id == selected_node:
            marker_color.append("#F39C12")
        elif stage == selected_stage:
            marker_color.append("#2AA198")
        else:
            marker_color.append("#5D6D7E")

        path = str(row.get("path", ""))
        tactical = row.get("tactical_scheme", "N/A")
        hover_text.append(
            "<br>".join(
                [
                    f"<b>Node {node_id}</b>",
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
        title="Scenario Tree Map (Horizontal Flow)",
        width=max(950, 240 * max(1, len(stage_order))),
        height=max(280, 120 + 32 * max(1, len([nid for nid in y_coords if nid in df["node_id"].tolist()]))),
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


def main() -> None:
    st.set_page_config(page_title="Squad Field Dashboard", page_icon="⚽", layout="wide")

    st.title("Squad Interactive Field")
    st.caption(
        "Interactive pitch view with starters, reserves, finance balance, and OVR evolution per window."
    )

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

    st.sidebar.caption(f"Active dataset: {selected_mode}")

    decisions = enrich_with_evolution(decisions, tree_meta)
    prepared_tree_meta = _prepare_tree_meta(tree_meta) if is_stochastic else pd.DataFrame()

    windows = sorted(decisions["window"].unique().tolist())
    time_label = "Stage" if is_stochastic else "Window"
    selected_window = st.slider(time_label, min_value=min(windows), max_value=max(windows), value=min(windows), step=1)

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
            selected_label = st.selectbox("Scenario Node", options=node_labels, index=selected_index)
            selected_node = int(node_map[selected_label])
            st.session_state[state_key] = selected_node
            node_row = stage_meta[stage_meta["node_id"] == selected_node].iloc[0]
        else:
            selected_node = int(st.selectbox("Scenario Node", options=nodes_in_window, index=0))

        if not prepared_tree_meta.empty and {"node_id", "parent_id", "stage"}.issubset(prepared_tree_meta.columns):
            st.subheader("Scenario Map")
            st.caption("Horizontal tree of stochastic branches. Nodes are sized by cumulative probability.")
            tree_fig = build_scenario_tree_figure(prepared_tree_meta, selected_node, selected_window)
            tree_key = f"scenario_tree_stage_{selected_window}"
            try:
                st.plotly_chart(
                    tree_fig,
                    width="content",
                    key=tree_key,
                    on_select="rerun",
                    selection_mode="points",
                )
                clicked_node = _extract_clicked_node(st.session_state.get(tree_key), valid_nodes=set(nodes_in_window))
                if clicked_node is not None and clicked_node != selected_node:
                    st.session_state[f"selected_node_stage_{selected_window}"] = clicked_node
                    st.rerun()
            except TypeError:
                st.plotly_chart(tree_fig, width="content")

    window_df = decisions[decisions["window"] == selected_window].copy()
    if selected_node is not None:
        window_df = window_df[window_df["node_id"] == selected_node].copy()

    if window_df.empty:
        st.warning("No data found for the selected window.")
        return

    scheme = window_df["formation_scheme"].dropna().iloc[0] if "formation_scheme" in window_df and not window_df["formation_scheme"].dropna().empty else "N/A"
    finance = summarize_window_finance(window_df, budget, selected_window, selected_node)

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
            st.caption(
                f"Node type: {'Leaf' if is_leaf else 'Internal'} | Children: {children_count} | Tactical scheme: {tactical} | Path: {path}"
            )

            if {"stage", "node_id", "cumulative_probability"}.issubset(prepared_tree_meta.columns):
                stage_nodes = prepared_tree_meta[prepared_tree_meta["stage"] == node_row.get("stage")][["node_id", "cumulative_probability"]].copy()
                stage_nodes = stage_nodes.sort_values(by="cumulative_probability", ascending=False)
                st.dataframe(stage_nodes, width="stretch", hide_index=True)

            if {"parent_id", "stage"}.issubset(prepared_tree_meta.columns):
                sibling_ids = _get_sibling_node_ids(prepared_tree_meta, selected_node)

                if sibling_ids:
                    siblings = prepared_tree_meta[prepared_tree_meta["node_id"].isin(sibling_ids)].copy()
                    sibling_options = [_build_node_label(row) for _, row in siblings.iterrows()]
                    sibling_map = dict(zip(sibling_options, siblings["node_id"].astype(int).tolist()))
                    st.subheader("Sibling Differences (Contingency Planning)")
                    selected_sibling_label = st.selectbox(
                        "Compare selected node against sibling",
                        options=sibling_options,
                        index=0,
                    )
                    sibling_node = int(sibling_map[selected_sibling_label])

                    sibling_df = decisions[
                        (decisions["window"] == selected_window)
                        & (decisions["node_id"] == sibling_node)
                    ].copy()

                    selected_squad = window_df.copy()
                    sibling_squad = sibling_df.copy()

                    diff_tbl = build_sibling_diff_table(selected_squad, sibling_squad)
                    if diff_tbl.empty:
                        st.info("No decision-state differences between selected node and chosen sibling.")
                    else:
                        st.caption(
                            "Green: bought/kept in selected branch | Red: sold/dispensed in selected branch"
                        )
                        st.dataframe(_style_sibling_diff(diff_tbl), width="stretch", hide_index=True)

    pitch_fig = build_pitch_figure(window_df, finance, selected_window)
    st.plotly_chart(pitch_fig, width="stretch")

    starters_tbl, reserves_tbl = build_window_tables(window_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Starting XI")
        st.dataframe(starters_tbl, width="stretch", hide_index=True)
    with col2:
        st.subheader("Reserves")
        st.dataframe(reserves_tbl, width="stretch", hide_index=True)

    st.subheader("Tactical Constraint Check")
    form_window = formation[formation["window"] == selected_window].copy()
    if selected_node is not None and "node_id" in form_window.columns:
        form_window = form_window[form_window["node_id"] == selected_node].copy()

    if not form_window.empty:
        form_window = form_window.sort_values(by="pos_group")
        st.dataframe(form_window[["formation_scheme", "pos_group", "required_count", "actual_starters", "slack_titular"]], width="stretch", hide_index=True)
    else:
        st.info("No tactical diagnostics found for this window.")

    st.markdown("---")
    st.code("streamlit run analysis/streamlit_dashboard.py", language="bash")


if __name__ == "__main__":
    main()
