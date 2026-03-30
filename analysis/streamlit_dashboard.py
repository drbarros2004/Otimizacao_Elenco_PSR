from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
SQUAD_DECISIONS_PATH = OUTPUT_DIR / "squad_decisions.csv"
BUDGET_EVOLUTION_PATH = OUTPUT_DIR / "budget_evolution.csv"
FORMATION_DIAGNOSTICS_PATH = OUTPUT_DIR / "formation_diagnostics.csv"

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
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        str(path.relative_to(ROOT))
        for path in [SQUAD_DECISIONS_PATH, BUDGET_EVOLUTION_PATH, FORMATION_DIAGNOSTICS_PATH]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing output files: " + ", ".join(missing) + ". Run the optimization pipeline first."
        )

    decisions = pd.read_csv(SQUAD_DECISIONS_PATH)
    budget = pd.read_csv(BUDGET_EVOLUTION_PATH)
    formation = pd.read_csv(FORMATION_DIAGNOSTICS_PATH)

    decisions["window"] = decisions["window"].astype(int)
    budget["window"] = budget["window"].astype(int)
    formation["window"] = formation["window"].astype(int)

    for col in ["in_squad", "is_starter", "bought", "sold", "ovr"]:
        decisions[col] = decisions[col].fillna(0).astype(int)

    decisions["name"] = decisions["name"].fillna("Unknown")
    decisions["pos_group"] = decisions["pos_group"].fillna("UNK")
    if "origin_club" not in decisions.columns:
        decisions["origin_club"] = "Unknown"
    decisions["origin_club"] = decisions["origin_club"].fillna("Unknown")
    if "origin_league" not in decisions.columns:
        decisions["origin_league"] = ""
    decisions["origin_league"] = decisions["origin_league"].fillna("")

    return decisions, budget, formation


def enrich_with_evolution(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.copy()
    prev = df[["player_id", "window", "ovr"]].copy()
    prev["window"] = prev["window"] + 1
    prev = prev.rename(columns={"ovr": "ovr_prev"})
    df = df.merge(prev, on=["player_id", "window"], how="left")
    return df


def summarize_window_finance(window_df: pd.DataFrame, budget_df: pd.DataFrame, window: int) -> dict:
    row = budget_df[budget_df["window"] == window]
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


def main() -> None:
    st.set_page_config(page_title="Squad Field Dashboard", page_icon="⚽", layout="wide")

    st.title("Squad Interactive Field")
    st.caption(
        "Interactive pitch view with starters, reserves, finance balance, and OVR evolution per window."
    )

    try:
        decisions, budget, formation = load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run: julia --project=. main.jl")
        return

    decisions = enrich_with_evolution(decisions)

    windows = sorted(decisions["window"].unique().tolist())
    selected_window = st.slider("Window", min_value=min(windows), max_value=max(windows), value=min(windows), step=1)

    window_df = decisions[decisions["window"] == selected_window].copy()
    if window_df.empty:
        st.warning("No data found for the selected window.")
        return

    scheme = window_df["formation_scheme"].dropna().iloc[0] if "formation_scheme" in window_df and not window_df["formation_scheme"].dropna().empty else "N/A"
    finance = summarize_window_finance(decisions, budget, selected_window)

    left, mid, right = st.columns([1, 1, 1])
    left.metric("Scheme", scheme)
    mid.metric("Cash", _money_millions(finance["cash"]))
    right.metric("Deficit", _money_millions(finance["deficit"]))

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
    if not form_window.empty:
        form_window = form_window.sort_values(by="pos_group")
        st.dataframe(form_window[["formation_scheme", "pos_group", "required_count", "actual_starters", "slack_titular"]], width="stretch", hide_index=True)
    else:
        st.info("No tactical diagnostics found for this window.")

    st.markdown("---")
    st.code("streamlit run analysis/streamlit_dashboard.py", language="bash")


if __name__ == "__main__":
    main()
