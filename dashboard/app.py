"""evalforge dashboard.

A Streamlit UI over the pipeline state. Read-only against Postgres locally and
the committed SQLite snapshot on Streamlit Cloud (set EVALFORGE_FIXTURE_MODE=1
or just deploy without DATABASE_URL). Review-queue mutations are no-ops in
fixture mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import data
import streamlit as st

st.set_page_config(page_title="evalforge", page_icon="🧪", layout="wide")

st.title("🧪 evalforge")
st.caption(
    "Auto-curated eval dataset, mined from production LLM logs. "
    "A self-growing test set that catches regressions and rewards scaling improvements."
)

stats = data.overview_stats()
fresh = data.freshness_stats()
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Pipeline runs", int(stats["pipeline_runs"]))
c2.metric("Logs ingested", int(stats["total_logs"]))
c3.metric("Test cases", int(stats["total_test_cases"]))
c4.metric("In review queue", int(stats["in_review"]))
c5.metric("Eval runs", int(stats["eval_runs"]))
c6.metric("Fresh ≤30d", f"{fresh['pct_last_30d']}%")

run_id = data.latest_run_id()
if run_id is None:
    st.info("No pipeline runs yet — execute `scripts/run_pipeline.py` to start.")
    st.stop()

tabs = st.tabs(
    [
        "Clusters",
        "Coverage & growth",
        "Eval dataset",
        "Review queue",
        "Eval runs",
        "Regression diff",
        "Claude usage",
    ]
)

# ───────────────────────────── Clusters ─────────────────────────────
with tabs[0]:
    st.subheader(f"Clusters (run #{run_id})")
    clusters = data.clusters_for_run(run_id)
    st.dataframe(clusters, use_container_width=True, hide_index=True)

    scatter = data.cluster_scatter(run_id)
    if len(scatter):
        chart = (
            alt.Chart(scatter)
            .mark_circle(size=80, opacity=0.7)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                color=alt.Color("cluster_name:N", legend=alt.Legend(title="Cluster")),
                shape=alt.Shape("is_outlier:N", legend=alt.Legend(title="Outlier")),
                tooltip=["cluster_name", "feature", "user_prompt", "outlier_reasons"],
            )
            .properties(height=520, title="UMAP projection of all logs")
        )
        st.altair_chart(chart, use_container_width=True)

# ─────────────────────── Coverage & growth ───────────────────────
with tabs[1]:
    st.subheader("Coverage heatmap")
    st.caption("How well-covered each (cluster × difficulty) cell is in the eval dataset.")
    heat = data.coverage_heatmap()
    if len(heat):
        order_cat = (
            heat.groupby("category")["n"].sum().sort_values(ascending=False).index.tolist()
        )
        order_diff = ["simple", "moderate", "hard", "adversarial"]
        chart = (
            alt.Chart(heat)
            .mark_rect()
            .encode(
                x=alt.X("difficulty:N", sort=order_diff, title="Difficulty"),
                y=alt.Y("category:N", sort=order_cat, title="Cluster"),
                color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="Cases"),
                tooltip=["category", "difficulty", "n"],
            )
            .properties(height=420)
        )
        text = (
            alt.Chart(heat)
            .mark_text(color="black")
            .encode(
                x=alt.X("difficulty:N", sort=order_diff),
                y=alt.Y("category:N", sort=order_cat),
                text="n:Q",
            )
        )
        st.altair_chart(chart + text, use_container_width=True)

    st.subheader("Coverage gaps")
    st.caption("Clusters with the most logs but the least eval-dataset representation.")
    gaps = data.coverage_gaps()
    st.dataframe(gaps, use_container_width=True, hide_index=True)

    st.subheader("Dataset growth over time")
    growth = data.growth_over_time()
    if len(growth):
        if "cumulative" not in growth.columns:
            growth["cumulative"] = growth["added"].cumsum()
        chart = (
            alt.Chart(growth)
            .mark_area(line=True, opacity=0.4)
            .encode(
                x=alt.X("bucket:T", title="Time"),
                y=alt.Y("cumulative:Q", title="Cumulative test cases"),
                tooltip=["bucket", "added", "cumulative"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Freshness")
    cf1, cf2, cf3 = st.columns(3)
    cf1.metric("Total cases", int(fresh["total"]))
    cf2.metric("Added in last 7d", int(fresh["last_7d"]))
    cf3.metric("Added in last 30d", int(fresh["last_30d"]))

# ─────────────────────────── Eval dataset ───────────────────────────
with tabs[2]:
    st.subheader("Eval dataset")
    df = data.eval_dataset_table()
    if not len(df):
        st.info("Eval dataset is empty — run `scripts/run_labeling.py <run_id>`.")
    else:
        cat_filter = st.multiselect("Category", sorted(df["category"].dropna().unique()))
        diff_filter = st.multiselect("Difficulty", sorted(df["difficulty"].dropna().unique()))
        view = df
        if cat_filter:
            view = view[view["category"].isin(cat_filter)]
        if diff_filter:
            view = view[view["difficulty"].isin(diff_filter)]
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "user_prompt": st.column_config.TextColumn("user_prompt", width="large"),
                "golden_answer": st.column_config.TextColumn("golden_answer", width="large"),
            },
        )

        case_id = st.number_input(
            "Inspect test case ID",
            min_value=int(df["id"].min()),
            max_value=int(df["id"].max()),
            value=int(df["id"].iloc[0]),
            step=1,
        )
        case = data.query(
            "SELECT * FROM eval_dataset WHERE id = " + ("%(i)s" if not data.USE_FIXTURE else "?"),
            ({"i": int(case_id)} if not data.USE_FIXTURE else (int(case_id),)),
        )
        if len(case):
            row = case.iloc[0]
            st.markdown(f"**User prompt:** {row['user_prompt']}")
            st.markdown(f"**Golden answer:** {row['golden_answer']}")
            st.markdown(f"**must_contain:** `{row['must_contain']}`")
            st.markdown(f"**must_not_contain:** `{row['must_not_contain']}`")

# ─────────────────────────── Review queue ───────────────────────────
with tabs[3]:
    st.subheader("Review queue")
    st.caption(
        "Auto-labels flagged for human review (low confidence or borderline quality=3). "
        "Approve to add to eval_dataset, reject to drop."
    )
    if data.USE_FIXTURE:
        st.warning("Fixture mode — review actions are disabled (read-only snapshot).")
    df = data.review_queue()
    if not len(df):
        st.success("Review queue is empty.")
    else:
        for _, row in df.iterrows():
            with st.expander(
                f"#{int(row['log_id'])} · q={row['quality_score']} · "
                f"{row['difficulty']} · {row['expected_behavior']} · "
                f"conf={row['confidence']:.2f} · {row['review_status']}"
            ):
                st.markdown(f"**Prompt:** {row['user_prompt']}")
                st.markdown(f"**Original response:** {row['response_preview']}")
                # Show 3 nearest existing eval_dataset cases for context
                sim = data.similar_eval_cases(str(row["user_prompt"]), k=3)
                if len(sim):
                    st.caption("Similar already-curated cases (for context):")
                    st.dataframe(
                        sim[["id", "category", "difficulty", "user_prompt", "similarity"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                notes_key = f"notes_{int(row['log_id'])}"
                notes = st.text_area("Reviewer notes", key=notes_key, height=80)
                bcols = st.columns(3)
                if bcols[0].button("✓ Approve", key=f"ap_{int(row['log_id'])}", disabled=data.USE_FIXTURE):
                    if data.approve_label(int(row["log_id"])):
                        st.success("Added to eval_dataset.")
                        st.rerun()
                if bcols[1].button("✗ Reject", key=f"rj_{int(row['log_id'])}", disabled=data.USE_FIXTURE):
                    if data.reject_label(int(row["log_id"]), notes or None):
                        st.success("Rejected.")
                        st.rerun()
                if bcols[2].button("✎ Save notes", key=f"nt_{int(row['log_id'])}", disabled=data.USE_FIXTURE):
                    if data.edit_label_notes(int(row["log_id"]), notes):
                        st.success("Notes saved.")

# ─────────────────────────── Eval runs ───────────────────────────
with tabs[4]:
    st.subheader("Eval runs")
    runs = data.eval_runs_table()
    if not len(runs):
        st.info("No completed eval runs yet — run `scripts/run_eval.py`.")
    else:
        st.dataframe(runs, use_container_width=True, hide_index=True)

        # Pass-rate bar chart
        chart = (
            alt.Chart(runs)
            .mark_bar()
            .encode(
                x=alt.X("candidate_label:N", title="Candidate", sort="-y"),
                y=alt.Y("pass_rate:Q", title="Pass rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("candidate_label:N", legend=None),
                tooltip=["candidate_label", "candidate_model", "passed", "failed", "pass_rate"],
            )
            .properties(height=300, title="Pass rate by candidate")
        )
        st.altair_chart(chart, use_container_width=True)

        chosen = st.selectbox("Inspect a run", runs["id"].tolist(), format_func=lambda i: f"run #{i}")
        st.dataframe(
            data.eval_results_for_run(int(chosen)),
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────── Regression diff ───────────────────────────
with tabs[5]:
    st.subheader("Regression diff")
    runs = data.eval_runs_table()
    if len(runs) < 2:
        st.info("Need ≥2 completed eval runs to diff.")
    else:
        ids = runs["id"].tolist()
        baseline = st.selectbox("Baseline run", ids, index=len(ids) - 1)
        candidate = st.selectbox("Candidate run", ids, index=0)
        diff = data.eval_diff(int(baseline), int(candidate))
        if not len(diff):
            st.info("No overlapping test cases between these runs.")
        else:
            base_pass = diff["baseline_passed"].astype(bool)
            cand_pass = diff["candidate_passed"].astype(bool)
            new_failures = diff[base_pass & ~cand_pass]
            new_passes = diff[~base_pass & cand_pass]
            c1, c2 = st.columns(2)
            c1.metric("New failures", len(new_failures))
            c2.metric("New passes", len(new_passes))
            st.markdown("### New failures (regressions)")
            st.dataframe(
                new_failures[
                    ["test_case_id", "category", "difficulty", "user_prompt", "candidate_failure_reasons"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("### Score deltas")
            d2 = diff.copy()
            d2["delta"] = d2["candidate_score"] - d2["baseline_score"]
            chart = (
                alt.Chart(d2)
                .mark_bar()
                .encode(
                    x=alt.X("test_case_id:O", title="Test case"),
                    y=alt.Y("delta:Q", title="Score delta (cand − baseline)"),
                    color=alt.condition(
                        alt.datum.delta < 0, alt.value("#d62728"), alt.value("#2ca02c")
                    ),
                    tooltip=["test_case_id", "category", "delta", "candidate_failure_reasons"],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

# ─────────────────────────── Claude usage ───────────────────────────
with tabs[6]:
    st.subheader("Claude usage")
    df = data.call_log_summary()
    if not len(df):
        st.info("No Claude calls logged yet.")
    else:
        df["cache_hit_rate"] = (df["cache_hits"] / df["calls"]).round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Cache hits are free re-runs. Concurrency capped at "
            "EVALFORGE_MAX_CONCURRENCY (default 2)."
        )
