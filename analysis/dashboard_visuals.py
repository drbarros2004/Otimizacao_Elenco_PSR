import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_logic import (
    FIELD_X_SCALE,
    FIELD_X_MAX,
    RESERVE_PANEL_X0,
    RESERVE_PANEL_X1,
    RESERVE_LEFT_TEXT_X,
    RESERVE_OVR_X,
    RESERVE_EVO_X,
    PLOT_X_MAX,
    PITCH_FIG_HEIGHT,
    FORMATION_POSITION_COORDS,
    _coerce_bool_or_none,
    _safe_int,
    _safe_float,
    _money_millions,
    _money_thousands,
    _foreign_status_text,
    _sort_reserves_by_priority,
    _prepare_tree_meta,
    _compute_squad_profile_metrics,
    _format_profile_metric,
)


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


def _get_position_coords(formation_scheme: str | None) -> dict[str, list[tuple[float, float]]]:
    if formation_scheme is None:
        return FORMATION_POSITION_COORDS["433"]
    scheme_key = str(formation_scheme).strip()
    return FORMATION_POSITION_COORDS.get(scheme_key, FORMATION_POSITION_COORDS["433"])


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


def _injury_badge(row: pd.Series) -> str:
    injured = int(row.get("injured", 0)) == 1
    return "<b><span style='color:#D42424'>✚</span></b>" if injured else ""


def _reinforcement_badge(row: pd.Series) -> str:
    new_signing = int(row.get("is_new_reinforcement", 0)) == 1
    return "<span title='Novo Reforço' style='color:#0E9F6E; font-size:1.2em;'> 🖋</span>" if new_signing else ""


def _root_inherited_badge(row: pd.Series) -> str:
    return ""


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
        nationality_txt = str(row.get("nationality", "Unknown"))
        foreign_txt = "Yes" if _coerce_bool_or_none(row.get("is_foreign")) else "No"

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
                    f"Nationality: {nationality_txt}",
                    f"Foreign Player: {foreign_txt}",
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
        nationality_txt = str(row.get("nationality", "Unknown"))
        foreign_txt = "Yes" if _coerce_bool_or_none(row.get("is_foreign")) else "No"

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
                    f"Nationality: {nationality_txt}",
                    f"Foreign Player: {foreign_txt}",
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

        if "nationality" not in df.columns:
            df["nationality"] = "Unknown"
        df["nationality"] = df["nationality"].fillna("").astype(str).str.strip()
        df.loc[df["nationality"] == "", "nationality"] = "Unknown"

        if "is_foreign" not in df.columns:
            df["is_foreign"] = pd.NA
        df["foreign"] = df["is_foreign"].apply(_foreign_status_text)

    starters = starters.sort_values(by=["pos_group", "ovr"], ascending=[True, False])
    reserves = _sort_reserves_by_priority(reserves)

    starter_cols = ["name", "pos_group", "ovr", "ovr_delta", "nationality", "foreign", "market_value", "acquisition_cost"]
    reserve_cols = ["name", "pos_group", "ovr", "ovr_delta", "nationality", "foreign", "market_value"]

    starters = starters[starter_cols].rename(columns={"name": "player", "pos_group": "pos"})
    reserves = reserves[reserve_cols].rename(columns={"name": "player", "pos_group": "pos"})

    starters["market_value"] = starters["market_value"].apply(_money_millions)
    starters["acquisition_cost"] = starters["acquisition_cost"].apply(_money_millions)
    reserves["market_value"] = reserves["market_value"].apply(_money_millions)

    return starters, reserves


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


def render_sold_players_table(sold_players_df: pd.DataFrame) -> None:
    st.subheader("Saídas do Elenco")
    if sold_players_df.empty:
        st.info("Sem vendas neste nó.")
        return

    sold_cols = ["name", "pos_group", "ovr", "nationality", "is_foreign", "market_value", "wage"]
    sold_table = sold_players_df[[col for col in sold_cols if col in sold_players_df.columns]].copy()
    sold_table = sold_table.rename(
        columns={
            "name": "Jogador",
            "pos_group": "Pos",
            "ovr": "OVR",
            "nationality": "Nacionalidade",
            "is_foreign": "Estrangeiro",
            "market_value": "Valor de Venda",
            "wage": "Salário",
        }
    )
    if "Nacionalidade" in sold_table.columns:
        sold_table["Nacionalidade"] = sold_table["Nacionalidade"].fillna("").astype(str).str.strip()
        sold_table.loc[sold_table["Nacionalidade"] == "", "Nacionalidade"] = "Unknown"
    if "Estrangeiro" in sold_table.columns:
        sold_table["Estrangeiro"] = sold_table["Estrangeiro"].apply(_foreign_status_text)
    sold_table["Valor de Venda"] = pd.to_numeric(sold_table["Valor de Venda"], errors="coerce").fillna(0.0).apply(_money_millions)
    sold_table["Salário"] = pd.to_numeric(sold_table["Salário"], errors="coerce").fillna(0.0).apply(_money_thousands)
    st.dataframe(sold_table, width="stretch", hide_index=True)


def render_financial_metrics(finance: dict, current_payroll_eur: float, salary_cap_eur: float | None) -> None:
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


def render_sibling_diff(diff_tbl: pd.DataFrame) -> None:
    if diff_tbl.empty:
        st.info("Sem diferenças de decisão entre os nós comparados.")
    else:
        st.dataframe(_style_sibling_diff(diff_tbl), width="stretch", hide_index=True)


def render_main_metrics(scheme_label: str, finance: dict) -> None:
    left, mid, right = st.columns([1, 1, 1])
    left.metric("Scheme", scheme_label)
    mid.metric("Cash", _money_millions(finance["cash"]))
    right.metric("Deficit", _money_millions(finance["deficit"]))


def render_starters_and_reserves(starters_tbl: pd.DataFrame, reserves_tbl: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Starting XI")
        st.dataframe(starters_tbl, width="stretch", hide_index=True)
    with col2:
        st.subheader("Reserves")
        st.dataframe(reserves_tbl, width="stretch", hide_index=True)


def render_tactical_constraint_table(form_window: pd.DataFrame) -> None:
    st.subheader("Tactical Constraint Check")
    if not form_window.empty:
        form_window = form_window.sort_values(by="pos_group")
        st.dataframe(
            form_window[["formation_scheme", "pos_group", "required_count", "actual_starters", "slack_titular"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No tactical diagnostics found for this window.")
