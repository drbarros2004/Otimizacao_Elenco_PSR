import pandas as pd
import streamlit as st

from dashboard_logic import (
    _build_sibling_choices,
    _compute_node_payroll_eur,
    _compute_profile_scale_bounds,
    _estimate_salary_cap_eur,
    _get_sibling_node_ids,
    _prepare_tree_meta,
    _with_display_best_xi,
    _with_post_decision_squad,
    enrich_with_evolution,
    load_data,
    summarize_window_finance,
)
from dashboard_visuals import (
    _build_node_label,
    _extract_clicked_node,
    build_pitch_figure,
    build_scenario_tree_figure,
    build_sibling_diff_table,
    build_squad_profile_radar,
    build_window_tables,
    render_financial_metrics,
    render_main_metrics,
    render_sibling_diff,
    render_sold_players_table,
    render_starters_and_reserves,
    render_tactical_constraint_table,
)


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
    st.caption("Interactive pitch view with starters, reserves, finance balance, and OVR evolution per window.")
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
                (prepared_tree_meta["stage"] == selected_window)
                & (prepared_tree_meta["node_id"].isin(nodes_in_window))
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

    window_df = _with_post_decision_squad(window_df)
    window_df = _with_display_best_xi(window_df, form_window)

    if "formation_scheme" in window_df and not window_df["formation_scheme"].dropna().empty:
        scheme = window_df["formation_scheme"].dropna().iloc[0]
    else:
        scheme = "N/A"

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
            mini_tree_height = int(760 * 0.56)
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
                render_sold_players_table(sold_players_df)
            with col_financas:
                render_financial_metrics(finance, current_payroll_eur, salary_cap_eur)

        sibling_ids = _get_sibling_node_ids(prepared_tree_meta, selected_node)
        sibling_options: list[str] = []
        sibling_option_ids: list[int] = []
        sibling_map: dict[str, int] = {}
        sibling_df = pd.DataFrame()

        if sibling_ids:
            _, sibling_options, sibling_option_ids, sibling_map = _build_sibling_choices(prepared_tree_meta, sibling_ids)

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
                render_sibling_diff(diff_tbl)
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

    scheme_label = scheme if selected_node is None else f"{scheme} | Node {selected_node}"
    render_main_metrics(scheme_label, finance)

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
                stage_nodes = prepared_tree_meta[
                    prepared_tree_meta["stage"] == node_row.get("stage")
                ][["node_id", "cumulative_probability"]].copy()
                stage_nodes = stage_nodes.sort_values(by="cumulative_probability", ascending=False)
                st.dataframe(stage_nodes, width="stretch", hide_index=True)

    starters_tbl, reserves_tbl = build_window_tables(window_df)
    render_starters_and_reserves(starters_tbl, reserves_tbl)
    render_tactical_constraint_table(form_window)

    st.markdown("---")
    st.code("streamlit run analysis/streamlit_dashboard.py", language="bash")


if __name__ == "__main__":
    main()
