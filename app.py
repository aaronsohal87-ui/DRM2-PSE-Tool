import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from io import BytesIO

st.set_page_config(page_title="DRM2 PSE Heatmap", page_icon="🔧", layout="wide")
st.title("🔧 DRM2 PSE Heatmap")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PROCESS_ORDER = ["INDUCT", "STOW", "PICK", "DISPATCH"]
PROCESS_COLORS = {"INDUCT": "midnightblue", "STOW": "darkorange", "PICK": "darkgreen", "DISPATCH": "firebrick"}

CYCLE_ORDER = ["CYCLE_1", "HV_A", "ADHOC", "ADHOC_1", "RTS_1", "LQ_A"]
CYCLE_COLORS = {"CYCLE_1": "steelblue", "HV_A": "darkorange", "ADHOC": "darkgreen", "ADHOC_1": "teal", "RTS_1": "purple", "LQ_A": "firebrick"}

SHIFT_HOUR_MAP = {0:"NS",1:"NS",2:"NS",3:"NS",4:"NS",5:"NS",6:"NS",7:"NS",8:"NS",9:"NS",10:"AM",11:"AM",12:"AM",13:"AM",14:"PM",15:"PM",16:"PM",17:"PM",18:"PM",19:"PM",20:"PM",21:"PM",22:"PM",23:"NS"}
SHIFT_ORDER = ["NS", "AM", "PM"]
SHIFT_DEFINITIONS = {"NS": "23:45 – 09:45 (Night Sort)", "AM": "09:45 – 14:00 (Pick & Dispatch)", "PM": "14:00 – 23:45 (Dispatch & RELO)"}
SHIFT_COLORS = {"NS": "midnightblue", "AM": "darkorange", "PM": "darkgreen"}

REQUIRED_PSE_COLS = ["Date", "Process", "Category", "Scannable ID", "Effective (Y/N)"]
REQUIRED_SCC_COLS = ["Tracking ID", "Sort Zone", "Aisle", "Cluster"]

LABEL_MAX = 30
CHART = (7, 2.5)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def hour_to_shift(hour):
    if pd.isna(hour): return "Unknown"
    try: return SHIFT_HOUR_MAP.get(int(hour), "Unknown")
    except (ValueError, TypeError): return "Unknown"


def assign_shift_pse(row):
    eot = row.get("Exception Open DT")
    if pd.notna(eot):
        try: return hour_to_shift(eot.hour)
        except (AttributeError, TypeError): pass
    pss = row.get("PSS Event DT")
    if pd.notna(pss):
        try: return hour_to_shift(pss.hour)
        except (AttributeError, TypeError): pass
    return "Unknown"


def fmt_pct(num, denom):
    if denom == 0: return "0.0%"
    return f"{round(num / denom * 100, 1)}%"


def fmt_cost(val):
    try:
        if pd.isna(val): return "£0.00"
        return f"£{float(val):,.2f}"
    except (ValueError, TypeError): return "£0.00"


def trunc(labels, mx=LABEL_MAX):
    return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]


def safe_top(s):
    try:
        c = s.dropna().value_counts()
        return c.index[0] if len(c)>0 else "N/A"
    except Exception: return "N/A"


def get_date_range(df):
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if len(dates) > 0:
            s = dates.min().strftime("%d %b %Y")
            e = dates.max().strftime("%d %b %Y")
            return s if s == e else f"{s} – {e}"
    if "Exception Open DT" in df.columns:
        valid = df["Exception Open DT"].dropna()
        if len(valid) > 0:
            s = valid.min().strftime("%d %b %Y")
            e = valid.max().strftime("%d %b %Y")
            return s if s == e else f"{s} – {e}"
    return ""


def make_bar_horiz(data, title, color="steelblue", figsize_width=7, max_label=LABEL_MAX):
    if len(data) == 0: return plt.subplots(figsize=(figsize_width, 2))[0]
    h = max(2, len(data)*0.3)
    fig, ax = plt.subplots(figsize=(figsize_width, h))
    labs = trunc(data.index, max_label)
    ax.barh(labs, data.values, color=color)
    ax.invert_yaxis()
    max_val = data.values.max() if len(data.values) > 0 else 1
    ax.set_xlim(right=max_val * 1.18)
    for i, v in enumerate(data.values):
        ax.text(v + max_val*0.02, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def make_bar_shift(data, title):
    shifts = [s for s in SHIFT_ORDER if s in data.index or True]
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    max_val = max([data[s] for s in SHIFT_ORDER]) if any(data[s] > 0 for s in SHIFT_ORDER) else 1
    ax.set_ylim(top=max_val * 1.25)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def make_eff_bar(df, group_col, title, top_n=15, color_eff="#2ecc71", color_ineff="#e74c3c"):
    """Horizontal bar showing effective vs ineffective stacked by group."""
    grouped = df.groupby(group_col).agg(
        Total=("Scannable ID", "count"),
        Effective=("Is Effective", "sum")
    ).sort_values("Total", ascending=False).head(top_n)
    grouped["Ineffective"] = grouped["Total"] - grouped["Effective"]
    grouped["Eff Rate"] = (grouped["Effective"] / grouped["Total"] * 100).round(1)

    if len(grouped) == 0:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No data", ha="center")
        return fig

    h = max(2, len(grouped)*0.35)
    fig, ax = plt.subplots(figsize=(7, h))
    labs = trunc(grouped.index, LABEL_MAX)
    y = range(len(grouped))

    ax.barh(labs, grouped["Effective"].values, color=color_eff, label="Effective")
    ax.barh(labs, grouped["Ineffective"].values, left=grouped["Effective"].values, color=color_ineff, label="Ineffective")
    ax.invert_yaxis()

    max_val = grouped["Total"].max()
    ax.set_xlim(right=max_val * 1.25)
    for i, (tot, rate) in enumerate(zip(grouped["Total"].values, grouped["Eff Rate"].values)):
        ax.text(tot + max_val*0.02, i, f"{int(tot)} ({rate}%)", va="center", fontsize=7)

    ax.set_xlabel("Events", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_pse(df):
    """Clean and enrich the PSE DataFrame."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Parse date columns
    date_cols = ["Exception Open Time", "Resolution Time", "Planned Departure Time",
                 "Induct End", "PSS Event Time", "Shipment Status Datetime"]
    for col in date_cols:
        if col in df.columns:
            df[col + "_DT"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Friendly aliases
    if "Exception Open Time_DT" in df.columns:
        df["Exception Open DT"] = df["Exception Open Time_DT"]
    if "PSS Event Time_DT" in df.columns:
        df["PSS Event DT"] = df["PSS Event Time_DT"]
    if "Resolution Time_DT" in df.columns:
        df["Resolution DT"] = df["Resolution Time_DT"]

    # Resolution time
    if "Resolution time taken(min)" in df.columns:
        df["Resolution Min"] = pd.to_numeric(
            df["Resolution time taken(min)"].astype(str).str.replace(",", ""), errors="coerce")
    else:
        df["Resolution Min"] = float("nan")

    # Cost
    if "gross_concession" in df.columns:
        df["Cost (£)"] = pd.to_numeric(
            df["gross_concession"].astype(str).str.replace("[£$,]", "", regex=True), errors="coerce").fillna(0)
    else:
        df["Cost (£)"] = 0.0

    # Shift
    df["Shift"] = df.apply(assign_shift_pse, axis=1)

    # Effective
    df["Effective"] = df["Effective (Y/N)"].astype(str).str.strip().str.upper()
    df["Is Effective"] = df["Effective"] == "Y"

    # SLA
    if "SLA (Y/N)" in df.columns:
        df["SLA Met"] = df["SLA (Y/N)"].astype(str).str.strip().str.upper() == "Y"
    else:
        df["SLA Met"] = False

    # Problem Solver display name
    if "Problem_Solver" in df.columns:
        df["PS Display"] = df["Problem_Solver"].astype(str).str.replace("@amazon.com", "", regex=False).str.strip()
    else:
        df["PS Display"] = "Unknown"

    # Standardise
    if "Process" in df.columns:
        df["Process"] = df["Process"].astype(str).str.strip().str.upper()
    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.strip()
    if "Reason" in df.columns:
        df["Reason Clean"] = df["Reason"].astype(str).str.strip().replace({"NONE": "No Reason", "nan": "Unknown", "": "Unknown"})
    else:
        df["Reason Clean"] = "Unknown"

    # DEA miss
    if "dea_miss" in df.columns:
        df["DEA Miss"] = pd.to_numeric(df["dea_miss"], errors="coerce").fillna(0).astype(int)
    else:
        df["DEA Miss"] = 0

    return df


def clean_scc(df):
    """Clean SCC — keep location columns for merge."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Tracking ID" in df.columns:
        df["Tracking ID"] = df["Tracking ID"].astype(str).str.strip()
    keep = ["Tracking ID", "Sort Zone", "Aisle", "Cluster", "Package Length", "Package Width", "Package Height"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def merge_pse_scc(pse_df, scc_df):
    """Merge PSE with SCC on Scannable ID / Tracking ID."""
    pse = pse_df.copy()
    scc = clean_scc(scc_df.copy())
    pse["_merge_key"] = pse["Scannable ID"].astype(str).str.strip()
    scc["_merge_key"] = scc["Tracking ID"].astype(str).str.strip()
    merged = pse.merge(scc, on="_merge_key", how="left", suffixes=("", "_scc"))
    merged = merged.drop(columns=["_merge_key"], errors="ignore")
    return merged


def filter_uk_ids(df):
    """Filter to only UK-prefix Scannable IDs (SCC compatible). Returns filtered df and removed count."""
    mask = df["Scannable ID"].astype(str).str.startswith("UK")
    removed = (~mask).sum()
    return df[mask].copy(), removed


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_health_score(df, total):
    if total == 0: return 5, "🟡", "No data", []
    score = 10
    reasons = []

    # Effectiveness rate
    eff_rate = df["Is Effective"].sum() / total
    if eff_rate < 0.5:
        score -= 3; reasons.append(f"Effectiveness very low ({eff_rate*100:.0f}%)")
    elif eff_rate < 0.65:
        score -= 2; reasons.append(f"Effectiveness below target ({eff_rate*100:.0f}%)")
    elif eff_rate < 0.75:
        score -= 1; reasons.append(f"Effectiveness slightly low ({eff_rate*100:.0f}%)")

    # SLA compliance
    sla_rate = df["SLA Met"].sum() / total
    if sla_rate < 0.5:
        score -= 2; reasons.append(f"SLA compliance poor ({sla_rate*100:.0f}%)")
    elif sla_rate < 0.7:
        score -= 1; reasons.append(f"SLA compliance below target ({sla_rate*100:.0f}%)")

    # Concentration — are problems focused in few categories?
    cat_c = df["Category"].dropna().value_counts()
    if len(cat_c) >= 2:
        top2_pct = cat_c.head(2).sum() / cat_c.sum()
        if top2_pct > 0.85:
            score -= 2; reasons.append("Problems highly concentrated in 1-2 categories")
        elif top2_pct > 0.7:
            score -= 1; reasons.append("Problems somewhat concentrated")

    # DEA misses
    dea_total = df["DEA Miss"].sum()
    if dea_total >= 5:
        score -= 2; reasons.append(f"{int(dea_total)} DEA misses")
    elif dea_total >= 2:
        score -= 1; reasons.append(f"{int(dea_total)} DEA miss(es)")

    score = max(1, min(10, score))
    if score >= 8: color = "🟢"; label = "Good"
    elif score >= 5: color = "🟡"; label = "Needs attention"
    else: color = "🔴"; label = "Action required"
    return score, color, label, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def render_summary_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    # UK IDs for SCC — copy-ready section
    with st.expander("📋 UK Tracking IDs — Copy into SCC for location data", expanded=False):
        uk_ids = df[df["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].unique()
        if len(uk_ids) > 0:
            st.caption(f"**{len(uk_ids)} unique UK IDs** ready to paste into SCC (Ctrl+A → Ctrl+C from the box below).")
            st.caption("Non-UK IDs (CR...) already removed. Paste these into SCC → Export → re-upload here as SCC CSV for full location analysis.")
            ids_text = "\n".join(uk_ids)
            st.text_area("UK Tracking IDs:", value=ids_text, height=200, key="summary_ids_box")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button("⬇️ Download as TXT", ids_text, "UK_IDs_for_SCC.txt", "text/plain", key="summary_dl_ids")
            with col_dl2:
                # CSV format (one column) for easy paste
                st.download_button("⬇️ Download as CSV", pd.DataFrame({"Tracking ID": uk_ids}).to_csv(index=False),
                                   "UK_IDs_for_SCC.csv", "text/csv", key="summary_dl_ids_csv")
        else:
            st.info("No UK tracking IDs in current filtered data.")

    # Process breakdown
    with st.expander("📦 By Process", expanded=True):
        proc_data = df.groupby("Process").agg(
            Total=("Scannable ID", "count"),
            Effective=("Is Effective", "sum")
        ).reindex(PROCESS_ORDER, fill_value=0)
        proc_data["Ineffective"] = proc_data["Total"] - proc_data["Effective"]
        proc_data["Eff Rate"] = (proc_data["Effective"] / proc_data["Total"] * 100).round(1)
        proc_data["% of All"] = (proc_data["Total"] / total * 100).round(1)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(proc_data[["Total", "Effective", "Ineffective", "Eff Rate", "% of All"]].rename(
                columns={"Eff Rate": "Eff %", "% of All": "% of Total"}), use_container_width=True)
        with col2:
            # Pie chart
            proc_totals = proc_data["Total"]
            proc_totals = proc_totals[proc_totals > 0]
            if len(proc_totals) > 0:
                fig, ax = plt.subplots(figsize=(3, 2.5))
                colors = [PROCESS_COLORS.get(p, "gray") for p in proc_totals.index]
                ax.pie(proc_totals.values, labels=proc_totals.index, colors=colors,
                       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 7})
                ax.set_title(f"Events by Process ({dr})", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)

    # Category breakdown
    with st.expander("🏷️ By Category", expanded=True):
        st.pyplot(make_eff_bar(df, "Category", f"All Categories — Effective vs Ineffective ({dr})"))

    # Shift breakdown
    with st.expander("⏰ By Shift", expanded=True):
        shift_data = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID", "count"),
            Effective=("Is Effective", "sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        shift_data["Ineffective"] = shift_data["Total"] - shift_data["Effective"]
        shift_data["Eff Rate"] = (shift_data["Effective"] / shift_data["Total"] * 100).round(1)
        shift_data["Window"] = [SHIFT_DEFINITIONS.get(s, "") for s in shift_data.index]

        st.dataframe(shift_data[["Total", "Effective", "Ineffective", "Eff Rate", "Window"]].rename(
            columns={"Eff Rate": "Eff %"}), use_container_width=True)

        st.pyplot(make_bar_shift(
            df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0),
            f"Events by Shift ({dr})"))

    # Status breakdown
    with st.expander("📋 By Status (Outcome)"):
        status_data = df["Status"].dropna().value_counts()
        if len(status_data) > 0:
            st.pyplot(make_bar_horiz(status_data.head(12), f"Event Outcomes ({dr})", color="teal"))

    # Hours
    with st.expander("🕐 Hour of Day"):
        if "Exception Open DT" in df.columns:
            hours = df["Exception Open DT"].dropna().dt.hour
            if len(hours) > 0:
                hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors = [SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, "Unknown"), "gray") for h in range(24)]
                ax.bar(range(24), hour_counts.values, color=colors)
                for h in range(24):
                    v = hour_counts.values[h]
                    if v > 0: ax.text(h, v+0.1, str(int(v)), ha="center", fontsize=6)
                ax.set_xlabel("Hour of Day", fontsize=8)
                ax.set_ylabel("Events", fontsize=8)
                ax.set_title("Exception Open Time — Hour of Day", fontsize=9)
                ax.set_xticks(range(24))
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")
                peak = hour_counts.idxmax()
                st.info(f"Peak hour: **{peak}:00** ({int(hour_counts.max())} events) — {SHIFT_HOUR_MAP.get(peak, 'Unknown')} shift")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_locations_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    has_scc = df["Cluster"].notna().any() if "Cluster" in df.columns else False

    if not has_scc:
        st.warning("⚠️ No SCC data — location drill-down uses Route only.")
        if "Route" in df.columns:
            route_data = df["Route"].dropna().value_counts().head(15)
            if len(route_data) > 0:
                st.pyplot(make_eff_bar(df[df["Route"].notna()], "Route", f"Events by Route ({dr})"))
        else:
            st.info("No Route data available.")
        return

    # Cluster view
    with st.expander("📍 By Cluster", expanded=True):
        st.pyplot(make_eff_bar(df[df["Cluster"].notna()], "Cluster", f"Events by Cluster ({dr})"))

    # Aisle view
    with st.expander("🏷️ By Aisle"):
        if "Aisle" in df.columns and df["Aisle"].notna().any():
            st.pyplot(make_eff_bar(df[df["Aisle"].notna()], "Aisle", f"Events by Aisle ({dr})", top_n=20))
        else:
            st.info("No aisle data.")

    # Sort Zone view
    with st.expander("🗂️ By Sort Zone"):
        if "Sort Zone" in df.columns and df["Sort Zone"].notna().any():
            st.pyplot(make_eff_bar(df[df["Sort Zone"].notna()], "Sort Zone", f"Events by Sort Zone ({dr})"))
        else:
            st.info("No sort zone data.")

    # Location drill-down
    with st.expander("🔍 Location Drill-Down"):
        clusters = sorted(df["Cluster"].dropna().unique().tolist())
        if clusters:
            sel_cluster = st.selectbox("Select Cluster:", clusters, key="loc_drill_cluster")
            filt = df[df["Cluster"] == sel_cluster]
            eff_n = int(filt["Is Effective"].sum())
            ineff_n = len(filt) - eff_n

            st.write(f"**{len(filt)} events** in {sel_cluster} — Effective: {eff_n} ({fmt_pct(eff_n, len(filt))}) | Ineffective: {ineff_n}")

            # Aisles in this cluster
            if "Aisle" in filt.columns:
                ad = filt["Aisle"].dropna().value_counts()
                if len(ad) > 0:
                    st.pyplot(make_bar_horiz(ad, f"{sel_cluster} — By Aisle", color="steelblue"))

            # Categories in this cluster
            cat_d = filt["Category"].dropna().value_counts()
            if len(cat_d) > 0:
                st.markdown("**Categories in this cluster:**")
                st.pyplot(make_bar_horiz(cat_d, f"{sel_cluster} — By Category", color="purple"))

            # Shift in this cluster
            shift_d = filt[filt["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
            st.pyplot(make_bar_shift(shift_d, f"{sel_cluster} — By Shift"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PROBLEM SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_ps_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("👤 Ranked: Worst → Best (by Effectiveness %)", expanded=True):
        st.caption("Sorted by effectiveness rate — lowest (worst) at top. Only solvers with 3+ events shown for fairness.")
        ps_data = df.groupby("PS Display").agg(
            Total=("Scannable ID", "count"),
            Effective=("Is Effective", "sum"),
            SLA=("SLA Met", "sum"),
            Avg_Res=("Resolution Min", "mean"),
            Cost=("Cost (£)", "sum")
        )
        ps_data["Ineffective"] = ps_data["Total"] - ps_data["Effective"]
        ps_data["Eff Rate"] = (ps_data["Effective"] / ps_data["Total"] * 100).round(1)
        ps_data["SLA %"] = (ps_data["SLA"] / ps_data["Total"] * 100).round(1)
        ps_data["Avg Res (min)"] = ps_data["Avg_Res"].round(0)

        # Filter to 3+ events and sort worst → best
        meaningful = ps_data[ps_data["Total"] >= 3].sort_values("Eff Rate", ascending=True)

        if len(meaningful) > 0:
            # Add rank column
            ranked = meaningful.reset_index()
            ranked.index = range(1, len(ranked)+1)
            ranked.index.name = "Rank"
            ranked = ranked.rename(columns={"PS Display": "Problem Solver", "Eff Rate": "Eff %", "Avg Res (min)": "Avg Min"})

            # Colour code: red if <50%, orange if <avg, green if above avg
            avg_eff = meaningful["Eff Rate"].mean()
            st.markdown(f"**Average effectiveness: {avg_eff:.1f}%** (across {len(meaningful)} solvers with 3+ events)")

            display_cols = ["Problem Solver", "Total", "Effective", "Ineffective", "Eff %", "SLA %", "Avg Min"]
            st.dataframe(ranked[display_cols], use_container_width=True, height=min(600, 35*len(ranked)+38))
        else:
            st.info("No problem solvers with 3+ events.")

        # Also show the full list (including <3 events) collapsed
        if len(ps_data) > len(meaningful):
            low_volume = ps_data[ps_data["Total"] < 3].sort_values("Eff Rate", ascending=True)
            if len(low_volume) > 0:
                st.caption(f"_{len(low_volume)} solver(s) with <3 events excluded from ranking (too few to judge):_")
                with st.expander(f"Show {len(low_volume)} low-volume solvers"):
                    lv_display = low_volume[["Total", "Effective", "Ineffective", "Eff Rate"]].rename(
                        columns={"Eff Rate": "Eff %"}).reset_index().rename(columns={"PS Display": "Problem Solver"})
                    lv_display.index = range(1, len(lv_display)+1)
                    st.dataframe(lv_display, use_container_width=True)

    with st.expander("🔴 Flagged — Below Average Effectiveness"):
        meaningful_flag = ps_data[ps_data["Total"] >= 5]
        if len(meaningful_flag) >= 3:
            avg_eff_flag = meaningful_flag["Eff Rate"].mean()
            flagged = meaningful_flag[meaningful_flag["Eff Rate"] < avg_eff_flag - 10].sort_values("Eff Rate", ascending=True)
            if len(flagged) > 0:
                st.error(f"🚨 {len(flagged)} solver(s) more than 10pp below average ({avg_eff_flag:.0f}% avg):")
                flag_display = flagged[["Total", "Effective", "Ineffective", "Eff Rate", "SLA %"]].rename(
                    columns={"Eff Rate": "Eff %"})
                flag_display.insert(0, "Problem Solver", flagged.index)
                flag_display.index = range(1, len(flag_display)+1)
                st.dataframe(flag_display, use_container_width=True)
            else:
                st.success(f"✅ No solvers significantly below average ({avg_eff_flag:.0f}%).")
        else:
            st.info("Need 3+ problem solvers with 5+ events to flag outliers.")

    with st.expander("📊 By Process — Who handles what?"):
        st.caption("Shows each solver's workload and effectiveness broken down by process type.")
        ps_proc = df.groupby(["PS Display", "Process"]).agg(
            Total=("Scannable ID", "count"),
            Effective=("Is Effective", "sum")
        ).reset_index()
        ps_proc["Eff %"] = (ps_proc["Effective"] / ps_proc["Total"] * 100).round(1)

        # Pivot for readability
        pivot_total = ps_proc.pivot_table(index="PS Display", columns="Process", values="Total", fill_value=0)
        pivot_eff = ps_proc.pivot_table(index="PS Display", columns="Process", values="Eff %", fill_value=0)

        # Only show solvers with 3+ total events
        solver_totals = pivot_total.sum(axis=1)
        valid_solvers = solver_totals[solver_totals >= 3].index
        pivot_total = pivot_total.loc[valid_solvers].sort_values(by=pivot_total.columns.tolist(), ascending=True)

        if len(pivot_total) > 0:
            st.markdown("**Event count by solver × process:**")
            st.dataframe(pivot_total.astype(int), use_container_width=True)

            st.markdown("**Effectiveness % by solver × process:**")
            pivot_eff_display = pivot_eff.loc[valid_solvers].round(1)
            st.dataframe(pivot_eff_display, use_container_width=True)

    with st.expander("⏱️ Resolution Time Distribution"):
        res_data = df["Resolution Min"].dropna()
        if len(res_data) > 0:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            # Cap at 99th percentile for display
            cap = res_data.quantile(0.99)
            plot_data = res_data[res_data <= cap]
            ax.hist(plot_data, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.axvline(x=res_data.median(), color="red", linestyle="--", linewidth=1, label=f"Median: {res_data.median():.0f} min")
            ax.set_xlabel("Resolution Time (min)", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.set_title("Resolution Time Distribution", fontsize=9)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(f"Median: {res_data.median():.0f} min | Mean: {res_data.mean():.0f} min | 90th pctile: {res_data.quantile(0.9):.0f} min")
        else:
            st.info("No resolution time data.")

    with st.expander("📊 Effectiveness Rate Chart (worst → best)"):
        # Horizontal bar chart of eff rate, sorted worst to best
        ps_chart = ps_data[ps_data["Total"] >= 3].sort_values("Eff Rate", ascending=True)
        if len(ps_chart) > 0:
            h = max(2, len(ps_chart)*0.3)
            fig, ax = plt.subplots(figsize=(7, h))
            colors = ["#e74c3c" if r < 50 else "#f39c12" if r < 70 else "#2ecc71" for r in ps_chart["Eff Rate"].values]
            labs = trunc(ps_chart.index, LABEL_MAX)
            ax.barh(labs, ps_chart["Eff Rate"].values, color=colors)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            for i, (rate, total_events) in enumerate(zip(ps_chart["Eff Rate"].values, ps_chart["Total"].values)):
                ax.text(rate + 1, i, f"{rate}% ({int(total_events)} events)", va="center", fontsize=7)
            ax.axvline(x=ps_chart["Eff Rate"].mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Effectiveness %", fontsize=8)
            ax.set_title(f"Problem Solver Effectiveness — Worst → Best ({dr})", fontsize=9)
            ax.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("🔴 <50% | 🟠 50-70% | 🟢 >70% | Dashed line = average")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: TIME & CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_time_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("🔄 By Actual Cycle", expanded=True):
        if "Actual Cycle" in df.columns:
            cycle_data = df.groupby("Actual Cycle").agg(
                Total=("Scannable ID", "count"),
                Effective=("Is Effective", "sum")
            ).sort_values("Total", ascending=False)
            cycle_data["Ineffective"] = cycle_data["Total"] - cycle_data["Effective"]
            cycle_data["Eff Rate"] = (cycle_data["Effective"] / cycle_data["Total"] * 100).round(1)
            st.dataframe(cycle_data[["Total", "Effective", "Ineffective", "Eff Rate"]].rename(
                columns={"Eff Rate": "Eff %"}), use_container_width=True)

            cycle_counts = df["Actual Cycle"].dropna().value_counts()
            if len(cycle_counts) > 0:
                st.pyplot(make_bar_horiz(cycle_counts, f"Events by Actual Cycle ({dr})", color="teal"))
        else:
            st.info("No Actual Cycle column.")

    with st.expander("📅 By Planned Cycle"):
        if "Planned Cycle" in df.columns:
            pc_data = df.groupby("Planned Cycle").agg(
                Total=("Scannable ID", "count"),
                Effective=("Is Effective", "sum")
            ).sort_values("Total", ascending=False)
            pc_data["Ineffective"] = pc_data["Total"] - pc_data["Effective"]
            pc_data["Eff Rate"] = (pc_data["Effective"] / pc_data["Total"] * 100).round(1)
            st.dataframe(pc_data[["Total", "Effective", "Ineffective", "Eff Rate"]].rename(
                columns={"Eff Rate": "Eff %"}), use_container_width=True)

    with st.expander("⏰ Shift Breakdown"):
        st.caption("NS: 23:45–09:45 | AM: 09:45–14:00 | PM: 14:00–23:45")
        shift_eff = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID", "count"),
            Effective=("Is Effective", "sum"),
            SLA=("SLA Met", "sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        shift_eff["Ineffective"] = shift_eff["Total"] - shift_eff["Effective"]
        shift_eff["Eff %"] = (shift_eff["Effective"] / shift_eff["Total"] * 100).round(1)
        shift_eff["SLA %"] = (shift_eff["SLA"] / shift_eff["Total"] * 100).round(1)
        shift_eff["Window"] = [SHIFT_DEFINITIONS.get(s, "") for s in shift_eff.index]
        st.dataframe(shift_eff[["Total", "Effective", "Ineffective", "Eff %", "SLA %", "Window"]], use_container_width=True)

    with st.expander("🕐 Hour of Day (Exception Open Time)"):
        if "Exception Open DT" in df.columns:
            hours = df["Exception Open DT"].dropna().dt.hour
            if len(hours) > 0:
                hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors = [SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, "Unknown"), "gray") for h in range(24)]
                ax.bar(range(24), hour_counts.values, color=colors)
                for h in range(24):
                    v = hour_counts.values[h]
                    if v > 0: ax.text(h, v+0.1, str(int(v)), ha="center", fontsize=6)
                ax.set_xlabel("Hour", fontsize=8)
                ax.set_ylabel("Events", fontsize=8)
                ax.set_title("Exception Open Hour", fontsize=9)
                ax.set_xticks(range(24))
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: COST & DEA
# ═══════════════════════════════════════════════════════════════════════════════

def render_cost_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    tc = df["Cost (£)"].sum()
    dea_total = df["DEA Miss"].sum()
    events_with_cost = (df["Cost (£)"] > 0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Concession Cost", fmt_cost(tc))
    c2.metric("Events with Cost", events_with_cost)
    c3.metric("Avg Cost/Event", fmt_cost(tc / events_with_cost) if events_with_cost > 0 else "£0.00")
    c4.metric("DEA Misses", int(dea_total))

    with st.expander("💰 Cost by Category", expanded=True):
        cost_df = df[df["Cost (£)"] > 0]
        if len(cost_df) > 0:
            cat_cost = cost_df.groupby("Category").agg(
                Events=("Scannable ID", "count"),
                Cost=("Cost (£)", "sum")
            ).sort_values("Cost", ascending=False).reset_index()
            cat_cost["Avg/Event"] = (cat_cost["Cost"] / cat_cost["Events"]).apply(fmt_cost)
            cat_cost["% of Cost"] = (cat_cost["Cost"] / tc * 100).round(1)
            cat_cost["Cost"] = cat_cost["Cost"].apply(fmt_cost)
            cat_cost.index = range(1, len(cat_cost)+1)
            st.dataframe(cat_cost, use_container_width=True)
        else:
            st.info("No concession cost data.")

    with st.expander("💰 Cost by Process"):
        cost_proc = df.groupby("Process").agg(
            Events=("Scannable ID", "count"),
            Cost=("Cost (£)", "sum")
        ).sort_values("Cost", ascending=False).reset_index()
        cost_proc["Avg/Event"] = (cost_proc["Cost"] / cost_proc["Events"]).apply(fmt_cost)
        cost_proc["Cost"] = cost_proc["Cost"].apply(fmt_cost)
        cost_proc.index = range(1, len(cost_proc)+1)
        st.dataframe(cost_proc, use_container_width=True)

    with st.expander("💰 Concession Bucket Breakdown"):
        if "concession_bucket_l1" in df.columns:
            cb = df["concession_bucket_l1"].dropna().value_counts()
            if len(cb) > 0:
                st.pyplot(make_bar_horiz(cb, f"Concession Buckets ({dr})", color="crimson"))
                # Table with cost
                cb_cost = df[df["concession_bucket_l1"].notna()].groupby("concession_bucket_l1").agg(
                    Events=("Scannable ID", "count"), Cost=("Cost (£)", "sum")
                ).sort_values("Cost", ascending=False).reset_index()
                cb_cost["Cost"] = cb_cost["Cost"].apply(fmt_cost)
                cb_cost.index = range(1, len(cb_cost)+1)
                st.dataframe(cb_cost, use_container_width=True)
            else:
                st.info("No concession bucket data.")
        else:
            st.info("No concession_bucket_l1 column.")

    with st.expander("🎯 DEA Misses"):
        dea_events = df[df["DEA Miss"] > 0]
        if len(dea_events) > 0:
            st.error(f"🚨 {len(dea_events)} event(s) with DEA misses — total misses: {int(dea_total)}")
            if "dea_bucket" in df.columns:
                dea_b = dea_events["dea_bucket"].dropna().value_counts()
                if len(dea_b) > 0:
                    st.pyplot(make_bar_horiz(dea_b, "DEA Miss Buckets", color="darkred"))
            # Show the events
            dea_cols = [c for c in ["Scannable ID", "Process", "Category", "PS Display", "Shift", "dea_bucket", "Cost (£)"] if c in dea_events.columns]
            dea_display = dea_events[dea_cols].reset_index(drop=True)
            dea_display.index = range(1, len(dea_display)+1)
            st.dataframe(dea_display, use_container_width=True)
        else:
            st.success("✅ No DEA misses.")

    with st.expander("💰 Top 10 Most Expensive Events"):
        top = df.nlargest(10, "Cost (£)")
        top_cols = [c for c in ["Scannable ID", "Process", "Category", "Effective", "PS Display", "Cost (£)", "concession_bucket_l1"] if c in top.columns]
        out = top[top_cols].reset_index(drop=True)
        out.index = range(1, len(out)+1)
        st.dataframe(out, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: ANALYSIS & TREND
# ═══════════════════════════════════════════════════════════════════════════════

def render_analysis_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    view = st.selectbox("View:", ["🔬 Analysis", "📈 Trend"], key="at_view")

    if view == "🔬 Analysis":
        st.markdown("### 🔬 Analysis — Key Findings & Recommendations")
        st.warning("⚠️ Data-driven suggestions — use your own judgement and local knowledge.")
        actions = []

        with st.expander("📍 Location Concentration", expanded=True):
            has_scc = df["Cluster"].notna().any() if "Cluster" in df.columns else False
            if has_scc:
                cl_c = df["Cluster"].dropna().value_counts()
                if len(cl_c) >= 2:
                    top5 = cl_c.head(5)
                    top5_pct = round(top5.sum() / cl_c.sum() * 100, 1)
                    ineff_in_top5 = df[(df["Cluster"].isin(top5.index)) & (~df["Is Effective"])]["Scannable ID"].count()

                    if top5_pct > 70:
                        st.error(f"🎯 HIGHLY concentrated. Top 5 clusters = **{top5_pct}%** of events ({ineff_in_top5} ineffective).")
                    elif top5_pct > 50:
                        st.warning(f"⚠️ Moderately concentrated. Top 5 = **{top5_pct}%**.")
                    else:
                        st.success(f"✅ Fairly spread. Top 5 = {top5_pct}%.")

                    st.markdown("**Clusters to focus on (most events + lowest effectiveness):**")
                    for cluster_name, count in top5.items():
                        cluster_df = df[df["Cluster"] == cluster_name]
                        eff_r = cluster_df["Is Effective"].mean() * 100
                        top_cat = cluster_df["Category"].value_counts().index[0] if len(cluster_df["Category"].dropna()) > 0 else "Unknown"
                        st.markdown(f"- **{cluster_name}**: {int(count)} events, {eff_r:.0f}% effective — top issue: {top_cat}")
                    actions.append(f"Walk clusters {', '.join(top5.index[:3])} — {top5_pct}% of events")
            else:
                if "Route" in df.columns and df["Route"].notna().any():
                    rt_c = df["Route"].dropna().value_counts()
                    top5 = rt_c.head(5)
                    st.markdown("**Top 5 Routes (no SCC data — using Route):**")
                    for rt_name, count in top5.items():
                        st.markdown(f"- **{rt_name}**: {int(count)} events")

        with st.expander("📊 Category Analysis"):
            cat_eff = df.groupby("Category").agg(
                Total=("Scannable ID", "count"),
                Effective=("Is Effective", "sum")
            ).sort_values("Total", ascending=False)
            cat_eff["Eff Rate"] = (cat_eff["Effective"] / cat_eff["Total"] * 100).round(1)
            worst_cat = cat_eff[cat_eff["Total"] >= 5].sort_values("Eff Rate").head(3)
            if len(worst_cat) > 0:
                st.markdown("**Worst categories by effectiveness (min 5 events):**")
                for cat_name, row in worst_cat.iterrows():
                    st.markdown(f"- **{cat_name}**: {int(row['Total'])} events, **{row['Eff Rate']}% effective**")
                actions.append(f"Focus on '{worst_cat.index[0]}' — lowest effectiveness ({worst_cat.iloc[0]['Eff Rate']}%)")

        with st.expander("⏰ Shift Imbalance"):
            shift_counts = df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
            assigned = shift_counts.sum()
            if assigned >= 10:
                expected_per = assigned / 3
                worst_shift = shift_counts.idxmax()
                worst_count = int(shift_counts.max())
                over_by = int(worst_count - expected_per)
                st.markdown(f"**Busiest shift: {worst_shift}** — {worst_count} events (expected ~{int(expected_per)}, over by {over_by})")
                for s in SHIFT_ORDER:
                    diff = int(shift_counts[s] - expected_per)
                    marker = "🔴" if diff > expected_per*0.3 else "🟢"
                    st.markdown(f"  {marker} {s}: {int(shift_counts[s])} (expected ~{int(expected_per)}, {'+'if diff>=0 else ''}{diff})")

        with st.expander("👤 Problem Solver Outliers"):
            ps_data = df.groupby("PS Display").agg(
                Total=("Scannable ID", "count"),
                Effective=("Is Effective", "sum")
            )
            ps_data["Eff Rate"] = (ps_data["Effective"] / ps_data["Total"] * 100).round(1)
            meaningful = ps_data[ps_data["Total"] >= 5]
            if len(meaningful) >= 3:
                mu = meaningful["Eff Rate"].mean()
                sigma = meaningful["Eff Rate"].std()
                if sigma > 0:
                    low_performers = meaningful[meaningful["Eff Rate"] < mu - sigma]
                    if len(low_performers) > 0:
                        st.error(f"🚨 {len(low_performers)} solver(s) more than 1σ below average ({mu:.0f}%):")
                        for ps_name, row in low_performers.iterrows():
                            st.markdown(f"- **{ps_name}**: {int(row['Total'])} events, **{row['Eff Rate']}% effective**")
                            actions.append(f"Coach {ps_name} — {row['Eff Rate']}% eff. ({int(row['Total'])} events)")
                    else:
                        st.success(f"✅ All solvers within normal range (avg {mu:.0f}%).")

        if actions:
            st.markdown("---")
            st.markdown("#### 📋 Suggested Actions")
            st.caption("Based on data patterns. Prioritise using local knowledge.")
            for i, a in enumerate(actions, 1):
                st.markdown(f"**{i}.** {a}")

    else:
        # TREND
        st.markdown("### 📈 Week-over-Week Trend")
        st.warning("⚠️ Trends need 3-4+ weeks to be meaningful.")
        st.caption("Upload one PSE Dashboard CSV per week, OR type values manually.")

        trend_mode = st.selectbox("Input:", ["📝 Type values", "📂 Upload CSVs"], key="trend_mode")

        if trend_mode == "📝 Type values":
            num_weeks = st.slider("How many weeks?", 2, 12, 4, key="tw_n")
            weeks_data = []
            for i in range(num_weeks):
                with st.expander(f"Week {i+1}", expanded=(i < 2)):
                    wl = st.text_input("Label:", value=f"W{i+1}", key=f"tw_l{i}")
                    wt = st.number_input("Total events:", min_value=0, value=0, step=1, key=f"tw_t{i}")
                    we = st.number_input("Effective:", min_value=0, value=0, step=1, key=f"tw_e{i}")
                    if wt > 0:
                        weeks_data.append({"Week": wl, "Total": int(wt), "Effective": int(we), "Ineffective": int(wt - we)})
            _render_trend(weeks_data)

        else:
            num_files = st.slider("How many weeks?", 2, 12, 4, key="tf_n")
            weeks_data = []
            for i in range(num_files):
                col1, col2 = st.columns([1, 3])
                with col1:
                    wl = st.text_input("Label:", value=f"W{i+1}", key=f"tf_l{i}")
                with col2:
                    f_up = st.file_uploader(f"PSE CSV ({wl}):", type="csv", key=f"tf_f{i}")
                if f_up:
                    try:
                        wdf = pd.read_csv(f_up, encoding="utf-8-sig")
                        wdf_total = len(wdf)
                        wdf_eff = (wdf["Effective (Y/N)"].astype(str).str.strip().str.upper() == "Y").sum() if "Effective (Y/N)" in wdf.columns else 0
                        weeks_data.append({"Week": wl, "Total": wdf_total, "Effective": int(wdf_eff), "Ineffective": wdf_total - int(wdf_eff)})
                        st.caption(f"→ {wl}: {wdf_total} events, {wdf_eff} effective ({fmt_pct(wdf_eff, wdf_total)})")
                    except Exception as e:
                        st.error(f"Error: {e}")
            _render_trend(weeks_data)


def _render_trend(weeks_data):
    if len(weeks_data) >= 2:
        wdf = pd.DataFrame(weeks_data)
        wdf["Eff Rate"] = (wdf["Effective"] / wdf["Total"] * 100).round(1)

        # Total events line
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(wdf["Week"], wdf["Total"], marker="o", color="steelblue", linewidth=2, label="Total")
        ax.plot(wdf["Week"], wdf["Ineffective"], marker="s", color="#e74c3c", linewidth=1.5, label="Ineffective")
        for i, row in wdf.iterrows():
            ax.annotate(str(int(row["Total"])), xy=(row["Week"], row["Total"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="steelblue")
            ax.annotate(str(int(row["Ineffective"])), xy=(row["Week"], row["Ineffective"]), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=7, color="#e74c3c")
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Events", fontsize=8)
        ax.set_title("PSE Events — Weekly Trend", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Effectiveness rate line
        fig2, ax2 = plt.subplots(figsize=(7, 2.5))
        ax2.plot(wdf["Week"], wdf["Eff Rate"], marker="o", color="darkgreen", linewidth=2)
        for i, row in wdf.iterrows():
            ax2.annotate(f"{row['Eff Rate']}%", xy=(row["Week"], row["Eff Rate"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
        ax2.set_xlabel("Week", fontsize=8)
        ax2.set_ylabel("Effectiveness %", fontsize=8)
        ax2.set_title("Effectiveness Rate — Weekly Trend", fontsize=9)
        ax2.tick_params(labelsize=7)
        ax2.axhline(y=wdf["Eff Rate"].mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)

        # Direction
        first = wdf.iloc[0]["Eff Rate"]
        last = wdf.iloc[-1]["Eff Rate"]
        if last > first + 5:
            st.success(f"📈 **Improving!** {first}% → {last}%")
        elif last < first - 5:
            st.error(f"📉 **Getting worse.** {first}% → {last}%")
        else:
            st.info(f"➡️ **Stable.** {first}% → {last}%")

        st.dataframe(wdf, use_container_width=True)
        st.download_button("⬇️ Download trend", wdf.to_csv(index=False), "pse_trend.csv", "text/csv", key="dl_trend")
    elif len(weeks_data) == 1:
        st.info("Need 2+ weeks to show a trend.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def render_export_tab(df, total, dr):
    st.markdown("#### 💾 Export")

    # Clean export columns
    exclude = ["Exception Open Time_DT", "Resolution Time_DT", "Planned Departure Time_DT",
               "Induct End_DT", "PSS Event Time_DT", "Shipment Status Datetime_DT",
               "Exception Open DT", "PSS Event DT", "Resolution DT", "_merge_key"]
    clean_cols = [c for c in df.columns if c not in exclude]

    st.download_button("⬇️ Download filtered data (CSV)", df[clean_cols].to_csv(index=False),
                       "PSE_Filtered.csv", "text/csv", key="dl_csv")

    st.markdown("---")
    st.markdown("**📋 UK Tracking IDs (for SCC)**")
    st.caption("Copy these into SCC to get cluster/aisle/sort zone data. Non-UK IDs already removed.")
    uk_ids = df[df["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].unique()
    if len(uk_ids) > 0:
        ids_text = "\n".join(uk_ids)
        st.text_area("Tracking IDs:", value=ids_text, height=200, key="ids_box")
        st.download_button("⬇️ Download IDs as TXT", ids_text, "UK_IDs_for_SCC.txt", "text/plain", key="dl_ids")
        st.caption(f"{len(uk_ids)} UK IDs ready to paste into SCC.")
    else:
        st.info("No UK tracking IDs found.")


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def render_guide():
    st.markdown("### 📖 How to Use This Tool")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**What you need:**

| File | Where to get it | What it contains |
|------|----------------|-----------------|
| **PSE Dashboard CSV** | PSE Dashboard → Raw Data → Export CSV | Every problem-solve event |
| **SCC CSV** (optional) | SCC → paste UK Tracking IDs → Export | Physical location (cluster, aisle, sort zone) |

**Steps:**
1. Export your **PSE Dashboard** raw data CSV
2. Upload it here — the tool auto-removes non-UK IDs and gives you a clean list
3. (Optional) Copy those UK IDs into **SCC** → Export → Upload here as second file
4. Use **filters** to focus on Pick, Stow, Induct, or Dispatch
5. Toggle Effective / Ineffective / All
6. Check the **Locations** tab to find where to walk

**⚠️ Non-UK IDs (e.g. CR...) are automatically filtered out** so you can paste IDs straight into SCC.
""")
    with st.expander("📊 What each tab does"):
        st.markdown("""
| Tab | Purpose |
|-----|---------|
| 📊 **Summary** | Overview — by process, category, shift, hour |
| 📍 **Locations** | Where problems concentrate — cluster, aisle, sort zone |
| 👤 **Problem Solvers** | Who's effective, who needs coaching |
| ⏰ **Time & Cycles** | When problems happen — shift, cycle, hour |
| 💰 **Cost & DEA** | Financial impact — concessions, DEA misses |
| 🔬 **Analysis & Trend** | Findings + week-over-week tracking |
| 💾 **Export** | Download filtered data + UK IDs for SCC |
""")
    with st.expander("🎯 Recommendations Logic"):
        st.markdown("""
The tool identifies:
- **Where:** Which clusters/aisles have the most issues + lowest effectiveness
- **What:** Which category dominates in that location
- **Who:** Which problem solvers are handling those areas
- **When:** Which shift/cycle the problems occur

Example:
> 🔴 **Cluster CA_A197** — 12 STOW events, 33% effective
> Root cause: STOW_WRONG_AISLE_CLUSTER (75%)
> → Walk this aisle. Check bin labels.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

mode = st.radio("Mode:", ["📖 Guide", "Single Station"], horizontal=True, key="mode")

if mode == "📖 Guide":
    render_guide()

elif mode == "Single Station":
    st.caption("Upload PSE Dashboard CSV. Optionally add SCC for cluster/aisle drill-down.")

    # ─── FILE UPLOAD ──────────────────────────────────────────────────────────
    c_pse, c_scc = st.columns(2)
    with c_pse:
        pse_file = st.file_uploader("🔧 PSE Dashboard CSV", type="csv", key="pse")
    with c_scc:
        scc_file = st.file_uploader("📋 SCC CSV (optional)", type="csv", key="scc")

    if pse_file:
        # ─── READ PSE ────────────────────────────────────────────────────────
        try:
            pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
        except Exception as e:
            st.error(f"❌ Error reading PSE CSV: {e}"); st.stop()

        pse_miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
        if pse_miss:
            st.error(f"❌ PSE CSV missing columns: {pse_miss}"); st.stop()

        # ─── FILTER NON-UK IDs ───────────────────────────────────────────────
        original_count = len(pse_df)
        pse_df, removed_count = filter_uk_ids(pse_df)
        if removed_count > 0:
            st.info(f"🔒 Removed **{removed_count}** non-UK IDs (e.g. CR...) — {len(pse_df)} UK IDs kept for SCC compatibility.")

        # ─── READ SCC ────────────────────────────────────────────────────────
        scc_df = None
        if scc_file:
            try:
                scc_df = pd.read_csv(scc_file, encoding="utf-8-sig")
            except Exception as e:
                st.error(f"❌ Error reading SCC CSV: {e}")
                scc_df = None
            if scc_df is not None:
                scc_miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
                if scc_miss:
                    st.warning(f"⚠️ SCC missing: {scc_miss} — location drill-down limited.")

        # ─── CLEAN & MERGE ────────────────────────────────────────────────────
        df = clean_pse(pse_df)

        if scc_df is not None:
            df = merge_pse_scc(df, scc_df)
            matched = df["Cluster"].notna().sum() if "Cluster" in df.columns else 0
            st.success(f"✅ **{len(df)} PSE events** — SCC matched: {matched}/{len(df)}")
        else:
            for col in ["Cluster", "Aisle", "Sort Zone"]:
                if col not in df.columns:
                    df[col] = None
            st.success(f"✅ **{len(df)} PSE events** loaded (no SCC — Route data only)")

        # ─── FILTERS ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎛️ Filters")
        f1, f2, f3 = st.columns(3)

        with f1:
            procs = sorted(df["Process"].dropna().unique().tolist())
            sel_procs = st.multiselect("Process:", procs, default=procs, key="f_proc")
        with f2:
            eff_filter = st.radio("Effectiveness:", ["All", "Ineffective Only", "Effective Only"], horizontal=True, key="f_eff")
        with f3:
            cats = sorted(df["Category"].dropna().unique().tolist())
            sel_cats = st.multiselect("Category:", cats, default=cats, key="f_cat")

        # Apply
        filtered = df.copy()
        if sel_procs:
            filtered = filtered[filtered["Process"].isin(sel_procs)]
        else:
            st.warning("Select at least one process."); st.stop()

        if eff_filter == "Ineffective Only":
            filtered = filtered[~filtered["Is Effective"]]
        elif eff_filter == "Effective Only":
            filtered = filtered[filtered["Is Effective"]]

        if sel_cats:
            filtered = filtered[filtered["Category"].isin(sel_cats)]
        else:
            st.warning("Select at least one category."); st.stop()

        total = len(filtered)
        if total == 0:
            st.warning("No events match filters."); st.stop()

        # ─── HEADLINE METRICS ────────────────────────────────────────────────
        dr = get_date_range(filtered)
        eff_count = int(filtered["Is Effective"].sum())
        ineff_count = total - eff_count
        sla_count = int(filtered["SLA Met"].sum())
        total_cost = filtered["Cost (£)"].sum()
        avg_res = filtered["Resolution Min"].dropna()

        st.markdown("---")
        if dr:
            st.caption(f"📅 **{dr}** | {total} of {len(df)} events shown")

        # Health score
        score, color, label, score_reasons = compute_health_score(filtered, total)
        st.markdown(f"**Health Score: {color} {score}/10 — {label}**" + (f" ({', '.join(score_reasons)})" if score_reasons else ""))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Events", total)
        c2.metric("Effective", f"{eff_count} ({fmt_pct(eff_count, total)})")
        c3.metric("Ineffective", f"{ineff_count} ({fmt_pct(ineff_count, total)})")
        c4.metric("SLA Met", fmt_pct(sla_count, total))
        c5.metric("Cost", fmt_cost(total_cost))
        c6.metric("Avg Resolution", f"{avg_res.mean():.0f} min" if len(avg_res) > 0 else "N/A")

        # ─── TABS ────────────────────────────────────────────────────────────
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Summary", "📍 Locations", "👤 Problem Solvers",
            "⏰ Time & Cycles", "💰 Cost & DEA", "🔬 Analysis & Trend", "💾 Export"
        ])

        with t1: render_summary_tab(filtered, total, dr)
        with t2: render_locations_tab(filtered, total, dr)
        with t3: render_ps_tab(filtered, total, dr)
        with t4: render_time_tab(filtered, total, dr)
        with t5: render_cost_tab(filtered, total, dr)
        with t6: render_analysis_tab(filtered, total, dr)
        with t7: render_export_tab(filtered, total, dr)

    else:
        st.info("👆 Upload your PSE Dashboard CSV to get started.")
