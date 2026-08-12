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

def get_solver_shift(df):
    """Determine which shift a problem solver primarily works from their event times."""
    ps_shifts = df.groupby("PS Display")["Shift"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown")
    return ps_shifts.to_dict()

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

def get_date_range(df):
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if len(dates) > 0:
            s = dates.min().strftime("%d %b %Y")
            e = dates.max().strftime("%d %b %Y")
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
    ax.set_xlabel("Count", fontsize=8); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    max_val = max([data[s] for s in SHIFT_ORDER]) if any(data[s] > 0 for s in SHIFT_ORDER) else 1
    ax.set_ylim(top=max_val * 1.25)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8); ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_eff_bar(df, group_col, title, top_n=15):
    grouped = df.groupby(group_col).agg(
        Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
    ).sort_values("Total", ascending=False).head(top_n)
    grouped["Ineffective"] = grouped["Total"] - grouped["Effective"]
    grouped["Eff Rate"] = (grouped["Effective"] / grouped["Total"] * 100).round(1)
    if len(grouped) == 0:
        fig, ax = plt.subplots(figsize=(7, 2)); ax.text(0.5, 0.5, "No data", ha="center"); return fig
    h = max(2, len(grouped)*0.35)
    fig, ax = plt.subplots(figsize=(7, h))
    labs = trunc(grouped.index, LABEL_MAX)
    ax.barh(labs, grouped["Effective"].values, color="#2ecc71", label="Effective")
    ax.barh(labs, grouped["Ineffective"].values, left=grouped["Effective"].values, color="#e74c3c", label="Ineffective")
    ax.invert_yaxis()
    max_val = grouped["Total"].max()
    ax.set_xlim(right=max_val * 1.25)
    for i, (tot, rate) in enumerate(zip(grouped["Total"].values, grouped["Eff Rate"].values)):
        ax.text(tot + max_val*0.02, i, f"{int(tot)} ({rate}%)", va="center", fontsize=7)
    ax.set_xlabel("Events", fontsize=8); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_pse(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    # Parse dates
    for col in ["Exception Open Time", "Resolution Time", "Planned Departure Time", "Induct End", "PSS Event Time", "Shipment Status Datetime"]:
        if col in df.columns:
            df[col + "_DT"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if "Exception Open Time_DT" in df.columns: df["Exception Open DT"] = df["Exception Open Time_DT"]
    if "PSS Event Time_DT" in df.columns: df["PSS Event DT"] = df["PSS Event Time_DT"]
    if "Resolution Time_DT" in df.columns: df["Resolution DT"] = df["Resolution Time_DT"]
    # Resolution min
    if "Resolution time taken(min)" in df.columns:
        df["Resolution Min"] = pd.to_numeric(df["Resolution time taken(min)"].astype(str).str.replace(",", ""), errors="coerce")
    else:
        df["Resolution Min"] = float("nan")
    # Cost
    if "gross_concession" in df.columns:
        df["Cost (£)"] = pd.to_numeric(df["gross_concession"].astype(str).str.replace("[£$,]", "", regex=True), errors="coerce").fillna(0)
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
    # PS Display
    if "Problem_Solver" in df.columns:
        df["PS Display"] = df["Problem_Solver"].astype(str).str.replace("@amazon.com", "", regex=False).str.strip()
    else:
        df["PS Display"] = "Unknown"
    # Process + Category
    if "Process" in df.columns: df["Process"] = df["Process"].astype(str).str.strip().str.upper()
    if "Category" in df.columns: df["Category"] = df["Category"].astype(str).str.strip()
    # DEA
    if "dea_miss" in df.columns:
        df["DEA Miss"] = pd.to_numeric(df["dea_miss"], errors="coerce").fillna(0).astype(int)
    else:
        df["DEA Miss"] = 0
    # Reason
    if "Reason" in df.columns:
        df["Reason Clean"] = df["Reason"].astype(str).str.strip().replace({"NONE": "No Reason", "nan": "Unknown", "": "Unknown"})
    else:
        df["Reason Clean"] = "Unknown"
    return df

def clean_scc(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "Tracking ID" in df.columns:
        df["Tracking ID"] = df["Tracking ID"].astype(str).str.strip()
    keep = ["Tracking ID", "Sort Zone", "Aisle", "Cluster", "Package Length", "Package Width", "Package Height"]
    return df[[c for c in keep if c in df.columns]]

def merge_pse_scc(pse_df, scc_df):
    pse = pse_df.copy()
    scc = clean_scc(scc_df.copy())
    pse["_merge_key"] = pse["Scannable ID"].astype(str).str.strip()
    scc["_merge_key"] = scc["Tracking ID"].astype(str).str.strip()
    merged = pse.merge(scc, on="_merge_key", how="left", suffixes=("", "_scc"))
    return merged.drop(columns=["_merge_key"], errors="ignore")

def filter_uk_ids(df):
    mask = df["Scannable ID"].astype(str).str.strip().str.startswith("UK")
    return df[mask].copy(), (~mask).sum()


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_health_score(df, total):
    if total == 0: return 5, "🟡", "No data", []
    score = 10; reasons = []
    eff_rate = df["Is Effective"].sum() / total
    if eff_rate < 0.5: score -= 3; reasons.append(f"Eff. very low ({eff_rate*100:.0f}%)")
    elif eff_rate < 0.65: score -= 2; reasons.append(f"Eff. below target ({eff_rate*100:.0f}%)")
    elif eff_rate < 0.75: score -= 1; reasons.append(f"Eff. slightly low ({eff_rate*100:.0f}%)")
    sla_rate = df["SLA Met"].sum() / total
    if sla_rate < 0.5: score -= 2; reasons.append(f"SLA poor ({sla_rate*100:.0f}%)")
    elif sla_rate < 0.7: score -= 1; reasons.append(f"SLA below target ({sla_rate*100:.0f}%)")
    dea_total = df["DEA Miss"].sum()
    if dea_total >= 5: score -= 2; reasons.append(f"{int(dea_total)} DEA misses")
    elif dea_total >= 2: score -= 1; reasons.append(f"{int(dea_total)} DEA miss(es)")
    score = max(1, min(10, score))
    if score >= 8: color = "🟢"; label = "Good"
    elif score >= 5: color = "🟡"; label = "Needs attention"
    else: color = "🔴"; label = "Action required"
    return score, color, label, reasons


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def render_summary_tab(filtered, total, dr, df_all):
    if total == 0: st.warning("No data."); return

    # ─── UK IDs for SCC ──────────────────────────────────────────────────────
    with st.expander("📋 Copy Tracking IDs into SCC", expanded=False):
        st.markdown("""
**How to use:**
1. Click **Download TXT** below
2. Open the downloaded file → Select All (Ctrl+A) → Copy (Ctrl+C)
3. Go to **SCC** → Paste into the search/upload box
4. Export the SCC result as CSV
5. Come back here and upload that CSV as the SCC file

_Non-UK IDs already removed. IDs are deduplicated (one per line)._
""")
        uk_ids = sorted(filtered[filtered["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].astype(str).str.strip().unique())
        if len(uk_ids) > 0:
            ids_text = "\n".join(uk_ids)
            st.metric("UK Tracking IDs", f"{len(uk_ids)} unique")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download TXT (one per line)", ids_text, "UK_IDs_for_SCC.txt", "text/plain", key="dl_ids_txt")
            with col2:
                st.download_button("⬇️ Download CSV", pd.DataFrame({"Tracking ID": uk_ids}).to_csv(index=False), "UK_IDs_for_SCC.csv", "text/csv", key="dl_ids_csv")
            with st.expander(f"Preview ({len(uk_ids)} IDs)"):
                st.code(ids_text[:2000] + ("\n..." if len(ids_text) > 2000 else ""), language=None)
        else:
            st.warning("No UK tracking IDs in current filter.")

    # ─── Process breakdown ────────────────────────────────────────────────────
    with st.expander("📦 By Process", expanded=True):
        proc_data = filtered.groupby("Process").agg(
            Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
        ).reindex(PROCESS_ORDER, fill_value=0)
        proc_data["Ineffective"] = proc_data["Total"] - proc_data["Effective"]
        proc_data["Eff %"] = (proc_data["Effective"] / proc_data["Total"] * 100).round(1)
        proc_data["% of All"] = (proc_data["Total"] / total * 100).round(1)
        col1, col2 = st.columns([1, 1])
        with col1: st.dataframe(proc_data[["Total", "Effective", "Ineffective", "Eff %", "% of All"]], use_container_width=True)
        with col2:
            pt = proc_data["Total"]; pt = pt[pt > 0]
            if len(pt) > 0:
                fig, ax = plt.subplots(figsize=(3, 2.5))
                ax.pie(pt.values, labels=pt.index, colors=[PROCESS_COLORS.get(p, "gray") for p in pt.index],
                       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 7})
                ax.set_title(f"By Process ({dr})", fontsize=8); plt.tight_layout(); st.pyplot(fig)

    # ─── Category breakdown ───────────────────────────────────────────────────
    with st.expander("🏷️ By Category", expanded=True):
        st.pyplot(make_eff_bar(filtered, "Category", f"Categories — Effective vs Ineffective ({dr})"))

    # ─── Shift ────────────────────────────────────────────────────────────────
    with st.expander("⏰ By Shift"):
        shift_data = filtered[filtered["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        shift_data["Ineffective"] = shift_data["Total"] - shift_data["Effective"]
        shift_data["Eff %"] = (shift_data["Effective"] / shift_data["Total"] * 100).round(1)
        shift_data["Window"] = [SHIFT_DEFINITIONS.get(s, "") for s in shift_data.index]
        st.dataframe(shift_data[["Total", "Effective", "Ineffective", "Eff %", "Window"]], use_container_width=True)
        st.pyplot(make_bar_shift(filtered[filtered["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0), f"Events by Shift ({dr})"))

    # ─── Hour of Day ──────────────────────────────────────────────────────────
    with st.expander("🕐 Hour of Day"):
        if "Exception Open DT" in filtered.columns:
            hours = filtered["Exception Open DT"].dropna().dt.hour
            if len(hours) > 0:
                hour_counts = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                colors = [SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, "Unknown"), "gray") for h in range(24)]
                ax.bar(range(24), hour_counts.values, color=colors)
                for h in range(24):
                    v = hour_counts.values[h]
                    if v > 0: ax.text(h, v+0.1, str(int(v)), ha="center", fontsize=6)
                ax.set_xlabel("Hour", fontsize=8); ax.set_ylabel("Events", fontsize=8)
                ax.set_title("Exception Open Hour", fontsize=9); ax.set_xticks(range(24))
                ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
                st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")

    # ─── Status ───────────────────────────────────────────────────────────────
    with st.expander("📋 By Status"):
        if "Status" in filtered.columns:
            status_data = filtered["Status"].dropna().value_counts()
            if len(status_data) > 0:
                st.pyplot(make_bar_horiz(status_data.head(12), f"Outcomes ({dr})", color="teal"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_locations_tab(filtered, total, dr):
    if total == 0: st.warning("No data."); return
    has_scc = "Cluster" in filtered.columns and filtered["Cluster"].notna().any()

    if not has_scc:
        st.warning("⚠️ No SCC data uploaded — upload SCC CSV for cluster/aisle drill-down.")
        if "Route" in filtered.columns and filtered["Route"].notna().any():
            st.pyplot(make_eff_bar(filtered[filtered["Route"].notna()], "Route", f"Events by Route ({dr})", top_n=20))
        return

    with st.expander("📍 By Cluster", expanded=True):
        st.pyplot(make_eff_bar(filtered[filtered["Cluster"].notna()], "Cluster", f"Events by Cluster ({dr})"))

    with st.expander("🏷️ By Aisle"):
        if "Aisle" in filtered.columns and filtered["Aisle"].notna().any():
            st.pyplot(make_eff_bar(filtered[filtered["Aisle"].notna()], "Aisle", f"Events by Aisle ({dr})", top_n=20))

    with st.expander("🗂️ By Sort Zone"):
        if "Sort Zone" in filtered.columns and filtered["Sort Zone"].notna().any():
            st.pyplot(make_eff_bar(filtered[filtered["Sort Zone"].notna()], "Sort Zone", f"Events by Sort Zone ({dr})"))

    with st.expander("🔍 Cluster Drill-Down"):
        clusters = sorted(filtered["Cluster"].dropna().unique().tolist())
        if clusters:
            sel = st.selectbox("Select Cluster:", clusters, key="loc_drill")
            filt = filtered[filtered["Cluster"] == sel]
            eff_n = int(filt["Is Effective"].sum())
            st.write(f"**{len(filt)} events** in {sel} — Eff: {eff_n}/{len(filt)} ({fmt_pct(eff_n, len(filt))})")
            if "Aisle" in filt.columns and filt["Aisle"].notna().any():
                st.pyplot(make_bar_horiz(filt["Aisle"].dropna().value_counts(), f"{sel} — By Aisle"))
            st.pyplot(make_bar_horiz(filt["Category"].dropna().value_counts(), f"{sel} — By Category", color="purple"))
            st.pyplot(make_bar_shift(filt[filt["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0), f"{sel} — By Shift"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PROBLEM SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_ps_tab(filtered, total, dr, df_all):
    """
    PS tab uses df_all (process+category filtered, but NOT effectiveness-filtered)
    to calculate Eff%. This means the ranking stays correct regardless of whether
    the user picks 'Ineffective Only' or 'Effective Only' in the filter bar.
    """
    if total == 0: st.warning("No data."); return

    # Build PS stats from df_all (not affected by effectiveness filter)
    solver_shifts = get_solver_shift(df_all)

    ps_data = df_all.groupby("PS Display").agg(
        Total=("Scannable ID", "count"),
        Effective=("Is Effective", "sum"),
        SLA=("SLA Met", "sum"),
        Avg_Res=("Resolution Min", "mean"),
        Cost=("Cost (£)", "sum")
    )
    ps_data["Ineffective"] = ps_data["Total"] - ps_data["Effective"]
    ps_data["Eff %"] = (ps_data["Effective"] / ps_data["Total"] * 100).round(1)
    ps_data["SLA %"] = (ps_data["SLA"] / ps_data["Total"] * 100).round(1)
    ps_data["Avg Min"] = ps_data["Avg_Res"].round(0)
    ps_data["Shift"] = ps_data.index.map(lambda x: solver_shifts.get(x, "Unknown"))

    with st.expander("👤 Ranked: Worst → Best Effectiveness (3+ events)", expanded=True):
        st.caption("Ranked by Eff % (lowest = worst). Eff % is ALWAYS calculated from all data regardless of filter. Shift = their most common working shift.")
        meaningful = ps_data[ps_data["Total"] >= 3].sort_values("Eff %", ascending=True).copy()
        if len(meaningful) > 0:
            avg_eff = meaningful["Eff %"].mean()
            st.markdown(f"**Average: {avg_eff:.1f}%** across {len(meaningful)} solvers")
            ranked = meaningful[["Shift", "Total", "Effective", "Ineffective", "Eff %", "SLA %", "Avg Min", "Cost"]].copy()
            ranked["Cost"] = ranked["Cost"].apply(fmt_cost)
            ranked.index.name = "Problem Solver"
            ranked = ranked.reset_index()
            ranked.index = range(1, len(ranked)+1)
            ranked.index.name = "Rank"
            st.dataframe(ranked, use_container_width=True, height=min(700, 35*len(ranked)+38))

            # Low-volume solvers
            low_vol = ps_data[ps_data["Total"] < 3]
            if len(low_vol) > 0:
                st.caption(f"_{len(low_vol)} solver(s) with <3 events excluded from ranking._")
        else:
            st.info("No solvers with 3+ events.")

    with st.expander("📊 Effectiveness Chart (worst → best)", expanded=True):
        meaningful = ps_data[ps_data["Total"] >= 3].sort_values("Eff %", ascending=True)
        if len(meaningful) > 0:
            h = max(2, len(meaningful)*0.3)
            fig, ax = plt.subplots(figsize=(7, h))
            colors = ["#e74c3c" if r < 50 else "#f39c12" if r < 70 else "#2ecc71" for r in meaningful["Eff %"].values]
            labs = [f"{name} [{meaningful.loc[name, 'Shift']}]" for name in meaningful.index]
            labs = trunc(labs, 35)
            ax.barh(labs, meaningful["Eff %"].values, color=colors)
            ax.invert_yaxis(); ax.set_xlim(0, 105)
            for i, (rate, t) in enumerate(zip(meaningful["Eff %"].values, meaningful["Total"].values)):
                ax.text(rate + 1, i, f"{rate}% ({int(t)} events)", va="center", fontsize=7)
            ax.axvline(x=meaningful["Eff %"].mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xlabel("Effectiveness %", fontsize=8)
            ax.set_title(f"Problem Solvers — Worst → Best ({dr})", fontsize=9)
            ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            st.caption("🔴 <50% | 🟠 50-70% | 🟢 >70% | ── average | [NS/AM/PM] = shift")

    with st.expander("🔴 Flagged — Need Coaching"):
        meaningful = ps_data[ps_data["Total"] >= 5].copy()
        if len(meaningful) >= 3:
            avg_eff = meaningful["Eff %"].mean()
            flagged = meaningful[meaningful["Eff %"] < avg_eff - 10].sort_values("Eff %", ascending=True)
            if len(flagged) > 0:
                st.error(f"🚨 {len(flagged)} solver(s) more than 10pp below average ({avg_eff:.0f}%):")
                for name, row in flagged.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']}]: {row['Eff %']}% effective ({int(row['Total'])} events, {int(row['Ineffective'])} ineffective)")
            else:
                st.success(f"✅ No solvers significantly below average ({avg_eff:.0f}%).")
        else:
            st.info("Need 3+ solvers with 5+ events.")

    with st.expander("📊 By Process — Who handles what?"):
        st.caption("Each solver's effectiveness by process type. Red = below 50%.")
        ps_proc = df_all.groupby(["PS Display", "Process"]).agg(
            Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
        ).reset_index()
        ps_proc["Eff %"] = (ps_proc["Effective"] / ps_proc["Total"] * 100).round(1)

        # Count pivot
        pivot_total = ps_proc.pivot_table(index="PS Display", columns="Process", values="Total", fill_value=0)
        solver_totals = pivot_total.sum(axis=1)
        valid = solver_totals[solver_totals >= 3].index
        if len(valid) > 0:
            st.markdown("**Event count:**")
            st.dataframe(pivot_total.loc[valid].sort_values(by=pivot_total.columns.tolist(), ascending=False).astype(int), use_container_width=True)
            st.markdown("**Effectiveness %:**")
            pivot_eff = ps_proc.pivot_table(index="PS Display", columns="Process", values="Eff %", fill_value=0)
            st.dataframe(pivot_eff.loc[valid].round(1), use_container_width=True)

    with st.expander("⏱️ Resolution Time"):
        res = df_all["Resolution Min"].dropna()
        if len(res) > 0:
            fig, ax = plt.subplots(figsize=(7, 2.5))
            cap = res.quantile(0.99)
            ax.hist(res[res <= cap], bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
            ax.axvline(x=res.median(), color="red", linestyle="--", linewidth=1, label=f"Median: {res.median():.0f} min")
            ax.set_xlabel("Minutes", fontsize=8); ax.set_ylabel("Count", fontsize=8)
            ax.set_title("Resolution Time Distribution", fontsize=9)
            ax.legend(fontsize=7); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
            st.caption(f"Median: {res.median():.0f}min | Mean: {res.mean():.0f}min | 90th: {res.quantile(0.9):.0f}min")

    with st.expander("🎯 Category × Solver — Where are they failing?"):
        st.caption("Shows which categories each solver is WORST at (lowest eff%). Only shows combos with 3+ events.")
        ps_cat = df_all.groupby(["PS Display", "Category"]).agg(
            Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
        ).reset_index()
        ps_cat["Eff %"] = (ps_cat["Effective"] / ps_cat["Total"] * 100).round(1)
        ps_cat_meaningful = ps_cat[ps_cat["Total"] >= 3].sort_values("Eff %", ascending=True).head(20)
        if len(ps_cat_meaningful) > 0:
            ps_cat_meaningful["Ineffective"] = ps_cat_meaningful["Total"] - ps_cat_meaningful["Effective"]
            display = ps_cat_meaningful[["PS Display", "Category", "Total", "Ineffective", "Eff %"]].rename(
                columns={"PS Display": "Solver"})
            display.index = range(1, len(display)+1)
            st.dataframe(display, use_container_width=True)
        else:
            st.info("Not enough data for category × solver breakdown.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: TIME & CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_time_tab(filtered, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("🔄 By Actual Cycle", expanded=True):
        if "Actual Cycle" in filtered.columns:
            cyc = filtered.groupby("Actual Cycle").agg(
                Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
            ).sort_values("Total", ascending=False)
            cyc["Ineffective"] = cyc["Total"] - cyc["Effective"]
            cyc["Eff %"] = (cyc["Effective"] / cyc["Total"] * 100).round(1)
            st.dataframe(cyc[["Total", "Effective", "Ineffective", "Eff %"]], use_container_width=True)
            st.pyplot(make_bar_horiz(filtered["Actual Cycle"].dropna().value_counts(), f"Actual Cycle ({dr})", color="teal"))

    with st.expander("📅 By Planned Cycle"):
        if "Planned Cycle" in filtered.columns:
            pc = filtered.groupby("Planned Cycle").agg(
                Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
            ).sort_values("Total", ascending=False)
            pc["Eff %"] = (pc["Effective"] / pc["Total"] * 100).round(1)
            st.dataframe(pc[["Total", "Effective", "Eff %"]], use_container_width=True)

    with st.expander("⏰ Shift Table"):
        st.caption("NS: 23:45–09:45 | AM: 09:45–14:00 | PM: 14:00–23:45")
        se = filtered[filtered["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID", "count"), Effective=("Is Effective", "sum"), SLA=("SLA Met", "sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        se["Eff %"] = (se["Effective"] / se["Total"] * 100).round(1)
        se["SLA %"] = (se["SLA"] / se["Total"] * 100).round(1)
        se["Window"] = [SHIFT_DEFINITIONS.get(s, "") for s in se.index]
        st.dataframe(se[["Total", "Effective", "Eff %", "SLA %", "Window"]], use_container_width=True)

    with st.expander("🕐 Hour of Day"):
        if "Exception Open DT" in filtered.columns:
            hours = filtered["Exception Open DT"].dropna().dt.hour
            if len(hours) > 0:
                hc = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.bar(range(24), hc.values, color=[SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h, ""), "gray") for h in range(24)])
                for h in range(24):
                    if hc.values[h] > 0: ax.text(h, hc.values[h]+0.1, str(int(hc.values[h])), ha="center", fontsize=6)
                ax.set_xlabel("Hour", fontsize=8); ax.set_ylabel("Events", fontsize=8)
                ax.set_xticks(range(24)); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: COST & DEA
# ═══════════════════════════════════════════════════════════════════════════════

def render_cost_tab(filtered, total, dr):
    if total == 0: st.warning("No data."); return
    tc = filtered["Cost (£)"].sum()
    events_with_cost = (filtered["Cost (£)"] > 0).sum()
    dea_total = int(filtered["DEA Miss"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Concession", fmt_cost(tc))
    c2.metric("Events with Cost", events_with_cost)
    c3.metric("Avg/Event", fmt_cost(tc/events_with_cost) if events_with_cost > 0 else "£0.00")
    c4.metric("DEA Misses", dea_total)

    with st.expander("💰 Cost by Category", expanded=True):
        cost_df = filtered[filtered["Cost (£)"] > 0]
        if len(cost_df) > 0:
            cc = cost_df.groupby("Category").agg(Events=("Scannable ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            cc["Avg"] = (cc["Cost"]/cc["Events"]).apply(fmt_cost); cc["Cost"] = cc["Cost"].apply(fmt_cost)
            cc.index = range(1,len(cc)+1); st.dataframe(cc, use_container_width=True)

    with st.expander("💰 Cost: Effective vs Ineffective"):
        st.caption("How much does ineffective PS cost? Compare concession cost when PS was effective vs not.")
        eff_cost = filtered[filtered["Is Effective"]]["Cost (£)"].sum()
        ineff_cost = filtered[~filtered["Is Effective"]]["Cost (£)"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Cost (Effective events)", fmt_cost(eff_cost))
        c2.metric("Cost (Ineffective events)", fmt_cost(ineff_cost))
        if ineff_cost > 0:
            st.warning(f"💡 **{fmt_cost(ineff_cost)}** in concessions on ineffective PS events — this is the cost of getting it wrong.")

    with st.expander("💰 Concession Buckets"):
        if "concession_bucket_l1" in filtered.columns:
            cb = filtered["concession_bucket_l1"].dropna().value_counts()
            if len(cb) > 0:
                st.pyplot(make_bar_horiz(cb, "Concession Buckets", color="crimson"))

    with st.expander("🎯 DEA Misses"):
        dea_events = filtered[filtered["DEA Miss"] > 0]
        if len(dea_events) > 0:
            st.error(f"🚨 {len(dea_events)} event(s) with DEA misses")
            if "dea_bucket" in filtered.columns:
                db = dea_events["dea_bucket"].dropna().value_counts()
                if len(db) > 0: st.pyplot(make_bar_horiz(db, "DEA Buckets", color="darkred"))
            cols = [c for c in ["Scannable ID","Process","Category","PS Display","Shift","dea_bucket"] if c in dea_events.columns]
            st.dataframe(dea_events[cols].reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No DEA misses.")

    with st.expander("💰 Top 10 Most Expensive"):
        top = filtered.nlargest(10, "Cost (£)")
        cols = [c for c in ["Scannable ID","Process","Category","Effective","PS Display","Cost (£)","concession_bucket_l1"] if c in top.columns]
        st.dataframe(top[cols].reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HOLES (NEW — finding gaps in PS)
# ═══════════════════════════════════════════════════════════════════════════════

def render_holes_tab(filtered, total, dr, df_all):
    if total == 0: st.warning("No data."); return
    st.markdown("### 🕳️ Finding Holes in Problem Solve")
    st.caption("This tab identifies systemic issues — things that keep going wrong and aren't being fixed.")

    with st.expander("🔁 Repeat Offender Packages", expanded=True):
        st.caption("Same tracking ID problem-solved multiple times = nobody actually fixed it the first time.")
        id_counts = df_all.groupby("Scannable ID").size()
        repeats = id_counts[id_counts > 1].sort_values(ascending=False)
        if len(repeats) > 0:
            st.error(f"🚨 **{len(repeats)} packages** were problem-solved more than once!")
            repeat_df = df_all[df_all["Scannable ID"].isin(repeats.index)].sort_values(["Scannable ID", "Exception Open DT"])
            cols = [c for c in ["Scannable ID", "Process", "Category", "Effective", "PS Display", "Shift", "Status"] if c in repeat_df.columns]
            st.dataframe(repeat_df[cols].reset_index(drop=True), use_container_width=True, height=300)
            st.caption(f"Total repeat events: {len(repeat_df)} (across {len(repeats)} unique packages)")
        else:
            st.success("✅ No packages problem-solved more than once.")

    with st.expander("⏳ Unresolved / No Resolution Time"):
        st.caption("Events where no resolution was recorded — may still be open or were abandoned.")
        unresolved = df_all[df_all["Resolution Min"].isna()]
        if len(unresolved) > 0:
            st.warning(f"⚠️ **{len(unresolved)} events** ({fmt_pct(len(unresolved), len(df_all))}) have no resolution time.")
            ur_by_cat = unresolved["Category"].value_counts()
            if len(ur_by_cat) > 0:
                st.pyplot(make_bar_horiz(ur_by_cat.head(10), "Unresolved by Category", color="gray"))
            ur_by_ps = unresolved["PS Display"].value_counts().head(10)
            if len(ur_by_ps) > 0:
                st.markdown("**Who leaves events unresolved most:**")
                st.dataframe(ur_by_ps.reset_index().rename(columns={"index":"Solver","PS Display":"Solver","count":"Unresolved"}), use_container_width=True)
        else:
            st.success("✅ All events have resolution times.")

    with st.expander("🐌 Slow Resolution (>60 min)"):
        st.caption("Events that took more than 60 minutes to resolve — potential SLA risks.")
        slow = df_all[df_all["Resolution Min"] > 60].sort_values("Resolution Min", ascending=False)
        if len(slow) > 0:
            st.warning(f"⚠️ **{len(slow)} events** took >60 min to resolve.")
            cols = [c for c in ["Scannable ID", "Process", "Category", "PS Display", "Shift", "Resolution Min", "Effective"] if c in slow.columns]
            st.dataframe(slow[cols].head(20).reset_index(drop=True), use_container_width=True)
            # By solver
            slow_by_ps = slow.groupby("PS Display").agg(Slow_Events=("Scannable ID","count"), Avg_Min=("Resolution Min","mean")).sort_values("Slow_Events", ascending=False)
            slow_by_ps["Avg_Min"] = slow_by_ps["Avg_Min"].round(0)
            st.markdown("**Who has the most slow resolutions:**")
            st.dataframe(slow_by_ps.head(10), use_container_width=True)
        else:
            st.success("✅ No events over 60 minutes.")

    with st.expander("❌ SLA Misses — Who, What, When"):
        sla_miss = df_all[~df_all["SLA Met"]]
        if len(sla_miss) > 0:
            st.warning(f"⚠️ **{len(sla_miss)} SLA misses** ({fmt_pct(len(sla_miss), len(df_all))})")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**By Solver:**")
                sla_ps = sla_miss["PS Display"].value_counts().head(10)
                st.dataframe(sla_ps.reset_index().rename(columns={"PS Display":"Solver","count":"SLA Misses"}), use_container_width=True)
            with col2:
                st.markdown("**By Category:**")
                sla_cat = sla_miss["Category"].value_counts().head(10)
                st.dataframe(sla_cat.reset_index().rename(columns={"Category":"Category","count":"SLA Misses"}), use_container_width=True)
            st.markdown("**By Shift:**")
            sla_shift = sla_miss[sla_miss["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0)
            st.pyplot(make_bar_shift(sla_shift, "SLA Misses by Shift"))
        else:
            st.success("✅ No SLA misses.")

    with st.expander("📉 Ineffective + High Cost (worst outcomes)"):
        st.caption("Events that were BOTH ineffective AND had a concession cost — the worst outcomes.")
        bad = df_all[(~df_all["Is Effective"]) & (df_all["Cost (£)"] > 0)].sort_values("Cost (£)", ascending=False)
        if len(bad) > 0:
            st.error(f"🚨 **{len(bad)} events** were ineffective AND had a concession (total: {fmt_cost(bad['Cost (£)'].sum())})")
            cols = [c for c in ["Scannable ID", "Process", "Category", "PS Display", "Shift", "Cost (£)", "concession_bucket_l1"] if c in bad.columns]
            st.dataframe(bad[cols].head(20).reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No ineffective events with concession cost.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: ANALYSIS & TREND
# ═══════════════════════════════════════════════════════════════════════════════

def render_analysis_tab(filtered, total, dr, df_all):
    if total == 0: st.warning("No data."); return
    view = st.selectbox("View:", ["🔬 Analysis", "📈 Trend"], key="at_view")

    if view == "🔬 Analysis":
        st.markdown("### 🔬 Key Findings & Recommended Actions")
        st.warning("⚠️ Data-driven suggestions — apply local judgement.")
        actions = []

        with st.expander("📍 Location Concentration", expanded=True):
            has_scc = "Cluster" in df_all.columns and df_all["Cluster"].notna().any()
            if has_scc:
                cl_c = df_all["Cluster"].dropna().value_counts()
                if len(cl_c) >= 2:
                    top5 = cl_c.head(5)
                    top5_pct = round(top5.sum() / cl_c.sum() * 100, 1)
                    if top5_pct > 70: st.error(f"🎯 Top 5 clusters = **{top5_pct}%** of events.")
                    elif top5_pct > 50: st.warning(f"⚠️ Top 5 = **{top5_pct}%**.")
                    else: st.success(f"✅ Fairly spread. Top 5 = {top5_pct}%.")
                    for name, count in top5.items():
                        cdf = df_all[df_all["Cluster"]==name]
                        eff_r = cdf["Is Effective"].mean()*100
                        top_cat = cdf["Category"].value_counts().index[0] if len(cdf["Category"].dropna())>0 else "?"
                        st.markdown(f"- **{name}**: {int(count)} events, {eff_r:.0f}% eff — top: {top_cat}")
                    actions.append(f"Walk {', '.join(top5.index[:3])}")
            else:
                st.info("Upload SCC for location analysis.")

        with st.expander("📊 Category Effectiveness"):
            cat_eff = df_all.groupby("Category").agg(Total=("Scannable ID","count"), Effective=("Is Effective","sum")).sort_values("Total", ascending=False)
            cat_eff["Eff %"] = (cat_eff["Effective"]/cat_eff["Total"]*100).round(1)
            worst = cat_eff[cat_eff["Total"]>=5].sort_values("Eff %").head(3)
            if len(worst)>0:
                st.markdown("**Worst categories (5+ events):**")
                for name, row in worst.iterrows():
                    st.markdown(f"- **{name}**: {row['Eff %']}% eff ({int(row['Total'])} events)")
                actions.append(f"Focus on '{worst.index[0]}' — {worst.iloc[0]['Eff %']}% eff")

        with st.expander("⏰ Shift Analysis"):
            sc = df_all[df_all["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
                Total=("Scannable ID","count"), Effective=("Is Effective","sum")
            ).reindex(SHIFT_ORDER, fill_value=0)
            sc["Eff %"] = (sc["Effective"]/sc["Total"]*100).round(1)
            worst_shift = sc["Eff %"].idxmin()
            st.markdown(f"**Worst shift: {worst_shift}** — {sc.loc[worst_shift,'Eff %']}% effective")
            for s in SHIFT_ORDER:
                marker = "🔴" if sc.loc[s,"Eff %"] < 60 else "🟡" if sc.loc[s,"Eff %"] < 75 else "🟢"
                st.markdown(f"  {marker} {s}: {sc.loc[s,'Eff %']}% ({int(sc.loc[s,'Total'])} events)")

        if actions:
            st.markdown("---")
            st.markdown("#### 📋 Suggested Actions")
            for i, a in enumerate(actions, 1): st.markdown(f"**{i}.** {a}")

    else:
        st.markdown("### 📈 Week-over-Week Trend")
        st.caption("Upload one PSE CSV per week, or type values manually.")
        trend_mode = st.selectbox("Input:", ["📝 Type values", "📂 Upload CSVs"], key="trend_mode")
        if trend_mode == "📝 Type values":
            nw = st.slider("Weeks:", 2, 12, 4, key="tw_n")
            weeks = []
            for i in range(nw):
                with st.expander(f"Week {i+1}", expanded=(i<2)):
                    wl = st.text_input("Label:", value=f"W{i+1}", key=f"tw_l{i}")
                    wt = st.number_input("Total:", min_value=0, value=0, step=1, key=f"tw_t{i}")
                    we = st.number_input("Effective:", min_value=0, value=0, step=1, key=f"tw_e{i}")
                    if wt > 0: weeks.append({"Week":wl,"Total":int(wt),"Effective":int(we)})
            _render_trend(weeks)
        else:
            nf = st.slider("Weeks:", 2, 12, 4, key="tf_n")
            weeks = []
            for i in range(nf):
                c1, c2 = st.columns([1,3])
                with c1: wl = st.text_input("Label:", value=f"W{i+1}", key=f"tf_l{i}")
                with c2: fu = st.file_uploader(f"PSE CSV:", type="csv", key=f"tf_f{i}")
                if fu:
                    try:
                        wdf = pd.read_csv(fu, encoding="utf-8-sig")
                        wt = len(wdf)
                        we = int((wdf["Effective (Y/N)"].astype(str).str.strip().str.upper()=="Y").sum()) if "Effective (Y/N)" in wdf.columns else 0
                        weeks.append({"Week":wl,"Total":wt,"Effective":we})
                        st.caption(f"→ {wl}: {wt} events, {we} effective ({fmt_pct(we,wt)})")
                    except Exception as e: st.error(f"Error: {e}")
            _render_trend(weeks)

def _render_trend(weeks):
    if len(weeks) >= 2:
        wdf = pd.DataFrame(weeks)
        wdf["Ineffective"] = wdf["Total"] - wdf["Effective"]
        wdf["Eff %"] = (wdf["Effective"]/wdf["Total"]*100).round(1)
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(wdf["Week"], wdf["Total"], marker="o", color="steelblue", linewidth=2, label="Total")
        ax.plot(wdf["Week"], wdf["Ineffective"], marker="s", color="#e74c3c", linewidth=1.5, label="Ineffective")
        for _,r in wdf.iterrows():
            ax.annotate(str(int(r["Total"])), xy=(r["Week"],r["Total"]), xytext=(0,8), textcoords="offset points", ha="center", fontsize=7, color="steelblue")
        ax.set_xlabel("Week",fontsize=8); ax.set_ylabel("Events",fontsize=8)
        ax.set_title("PSE Events — Weekly",fontsize=9); ax.legend(fontsize=7)
        ax.tick_params(labelsize=7); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(7,2.5))
        ax2.plot(wdf["Week"], wdf["Eff %"], marker="o", color="darkgreen", linewidth=2)
        for _,r in wdf.iterrows():
            ax2.annotate(f"{r['Eff %']}%", xy=(r["Week"],r["Eff %"]), xytext=(0,8), textcoords="offset points", ha="center", fontsize=7)
        ax2.axhline(y=wdf["Eff %"].mean(), color="gray", linestyle="--", linewidth=1)
        ax2.set_xlabel("Week",fontsize=8); ax2.set_ylabel("Eff %",fontsize=8)
        ax2.set_title("Effectiveness Trend",fontsize=9); ax2.tick_params(labelsize=7)
        plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig2)

        f, l = wdf.iloc[0]["Eff %"], wdf.iloc[-1]["Eff %"]
        if l > f+5: st.success(f"📈 Improving: {f}% → {l}%")
        elif l < f-5: st.error(f"📉 Worsening: {f}% → {l}%")
        else: st.info(f"➡️ Stable: {f}% → {l}%")
        st.dataframe(wdf, use_container_width=True)
    elif len(weeks)==1: st.info("Need 2+ weeks.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def render_export_tab(filtered, total, dr):
    st.markdown("#### 💾 Export")
    exclude = [c for c in filtered.columns if c.endswith("_DT") or c == "_merge_key"]
    clean_cols = [c for c in filtered.columns if c not in exclude]
    st.download_button("⬇️ Download filtered data (CSV)", filtered[clean_cols].to_csv(index=False), "PSE_Filtered.csv", "text/csv", key="dl_csv")
    st.markdown("---")
    uk_ids = sorted(filtered[filtered["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].astype(str).str.strip().unique())
    if len(uk_ids) > 0:
        st.download_button("⬇️ UK IDs for SCC (TXT)", "\n".join(uk_ids), "UK_IDs.txt", "text/plain", key="exp_ids")


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def render_guide():
    st.markdown("### 📖 How to Use This Tool")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**What you need:**

| File | Source | Contains |
|------|--------|----------|
| **PSE Dashboard CSV** | PSE Dashboard → Raw Data → Export | Every problem-solve event |
| **SCC CSV** (optional) | SCC → paste UK IDs → Export | Physical location (cluster, aisle, sort zone) |

**Steps:**
1. Export **PSE Dashboard** raw data CSV
2. Upload it here — non-UK IDs (CR...) auto-removed
3. Go to **Summary** tab → expand "📋 Copy Tracking IDs into SCC"
4. Download the TXT file → open it → Ctrl+A → Ctrl+C
5. Paste into **SCC** search → Export → Upload here as SCC CSV
6. Now all tabs have full location drill-down

**Filters:** Pick Process (Induct/Stow/Pick/Dispatch) + Effectiveness + Category
""")
    with st.expander("📊 Tab Guide"):
        st.markdown("""
| Tab | What it shows |
|-----|--------------|
| 📊 **Summary** | Overview + UK IDs for SCC copy |
| 📍 **Locations** | Worst clusters/aisles/sort zones |
| 👤 **Problem Solvers** | Ranked worst→best, with shift + process breakdown |
| ⏰ **Time & Cycles** | When problems happen |
| 💰 **Cost & DEA** | Financial impact |
| 🕳️ **Holes** | Repeat offenders, unresolved, slow, SLA misses, costly failures |
| 🔬 **Analysis & Trend** | Statistical findings + WoW tracking |
| 💾 **Export** | Download data |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

mode = st.radio("Mode:", ["📖 Guide", "Single Station"], horizontal=True, key="mode")

if mode == "📖 Guide":
    render_guide()

elif mode == "Single Station":
    st.caption("Upload PSE Dashboard CSV. Optionally add SCC for location drill-down.")

    c_pse, c_scc = st.columns(2)
    with c_pse: pse_file = st.file_uploader("🔧 PSE Dashboard CSV", type="csv", key="pse")
    with c_scc: scc_file = st.file_uploader("📋 SCC CSV (optional)", type="csv", key="scc")

    if pse_file:
        try: pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
        except Exception as e: st.error(f"❌ {e}"); st.stop()

        pse_miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
        if pse_miss: st.error(f"❌ Missing columns: {pse_miss}"); st.stop()

        # Filter non-UK IDs
        original_n = len(pse_df)
        pse_df, removed_n = filter_uk_ids(pse_df)
        if removed_n > 0:
            st.info(f"🔒 Removed {removed_n} non-UK IDs — {len(pse_df)} kept.")

        # Read SCC
        scc_df = None
        if scc_file:
            try: scc_df = pd.read_csv(scc_file, encoding="utf-8-sig")
            except Exception as e: st.error(f"❌ SCC error: {e}"); scc_df = None
            if scc_df is not None:
                miss = [c for c in REQUIRED_SCC_COLS if c not in scc_df.columns]
                if miss: st.warning(f"⚠️ SCC missing: {miss}")

        # Clean & Merge
        df = clean_pse(pse_df)
        if scc_df is not None:
            df = merge_pse_scc(df, scc_df)
            matched = df["Cluster"].notna().sum() if "Cluster" in df.columns else 0
            st.success(f"✅ **{len(df)} events** — SCC matched: {matched}/{len(df)}")
        else:
            for col in ["Cluster", "Aisle", "Sort Zone"]:
                if col not in df.columns: df[col] = None
            st.success(f"✅ **{len(df)} events** loaded (no SCC)")

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

        # Apply process + category filter (this is "df_all" for PS tab)
        df_proc_cat = df.copy()
        if sel_procs: df_proc_cat = df_proc_cat[df_proc_cat["Process"].isin(sel_procs)]
        else: st.warning("Select a process."); st.stop()
        if sel_cats: df_proc_cat = df_proc_cat[df_proc_cat["Category"].isin(sel_cats)]
        else: st.warning("Select a category."); st.stop()

        # Apply effectiveness filter for display
        filtered = df_proc_cat.copy()
        if eff_filter == "Ineffective Only": filtered = filtered[~filtered["Is Effective"]]
        elif eff_filter == "Effective Only": filtered = filtered[filtered["Is Effective"]]

        total = len(filtered)
        if total == 0: st.warning("No events match."); st.stop()

        # ─── METRICS ─────────────────────────────────────────────────────────
        dr = get_date_range(filtered)
        eff_count = int(filtered["Is Effective"].sum())
        ineff_count = total - eff_count
        sla_count = int(filtered["SLA Met"].sum())
        total_cost = filtered["Cost (£)"].sum()
        avg_res = filtered["Resolution Min"].dropna()

        st.markdown("---")
        if dr: st.caption(f"📅 **{dr}** | {total} of {len(df)} events shown")

        score, color, label, reasons = compute_health_score(df_proc_cat, len(df_proc_cat))
        st.markdown(f"**Health: {color} {score}/10 — {label}**" + (f" ({', '.join(reasons)})" if reasons else ""))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Events", total)
        c2.metric("Effective", f"{eff_count} ({fmt_pct(eff_count, total)})")
        c3.metric("Ineffective", f"{ineff_count} ({fmt_pct(ineff_count, total)})")
        c4.metric("SLA Met", fmt_pct(sla_count, total))
        c5.metric("Cost", fmt_cost(total_cost))
        c6.metric("Avg Res", f"{avg_res.mean():.0f}m" if len(avg_res)>0 else "N/A")

        # ─── TABS ────────────────────────────────────────────────────────────
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "📊 Summary", "📍 Locations", "👤 Problem Solvers",
            "⏰ Time & Cycles", "💰 Cost & DEA", "🕳️ Holes",
            "🔬 Analysis & Trend", "💾 Export"
        ])

        with t1: render_summary_tab(filtered, total, dr, df_proc_cat)
        with t2: render_locations_tab(filtered, total, dr)
        with t3: render_ps_tab(filtered, total, dr, df_proc_cat)
        with t4: render_time_tab(filtered, total, dr)
        with t5: render_cost_tab(filtered, total, dr)
        with t6: render_holes_tab(filtered, total, dr, df_proc_cat)
        with t7: render_analysis_tab(filtered, total, dr, df_proc_cat)
        with t8: render_export_tab(filtered, total, dr)
    else:
        st.info("👆 Upload your PSE Dashboard CSV to get started.")
