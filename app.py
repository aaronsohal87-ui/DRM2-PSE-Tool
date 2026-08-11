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
# CORE FUNCTIONS — INPUT & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def hour_to_shift(hour):
    """Map hour (0-23) to shift name."""
    if pd.isna(hour):
        return "Unknown"
    try:
        return SHIFT_HOUR_MAP.get(int(hour), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


def assign_shift_pse(row):
    """Assign shift based on Exception Open Time hour."""
    eot = row.get("Exception Open DT")
    if pd.notna(eot):
        try:
            return hour_to_shift(eot.hour)
        except (AttributeError, TypeError):
            pass
    # Fallback: try PSS Event Time
    pss = row.get("PSS Event DT")
    if pd.notna(pss):
        try:
            return hour_to_shift(pss.hour)
        except (AttributeError, TypeError):
            pass
    return "Unknown"


def clean_pse(df):
    """Clean and enrich the PSE DataFrame."""
    df = df.copy()

    # Standardise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Parse date columns
    date_cols_to_parse = [
        "Exception Open Time", "Resolution Time", "Planned Departure Time",
        "Induct End", "PSS Event Time", "Shipment Status Datetime"
    ]
    for col in date_cols_to_parse:
        if col in df.columns:
            df[col + "_DT"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Friendlier column aliases
    if "Exception Open Time_DT" in df.columns:
        df["Exception Open DT"] = df["Exception Open Time_DT"]
    if "PSS Event Time_DT" in df.columns:
        df["PSS Event DT"] = df["PSS Event Time_DT"]
    if "Resolution Time_DT" in df.columns:
        df["Resolution DT"] = df["Resolution Time_DT"]

    # Parse Resolution time taken (already numeric in sample, but be safe)
    if "Resolution time taken(min)" in df.columns:
        df["Resolution Min"] = pd.to_numeric(
            df["Resolution time taken(min)"].astype(str).str.replace(",", ""), errors="coerce"
        )
    else:
        df["Resolution Min"] = float("nan")

    # Parse gross_concession
    if "gross_concession" in df.columns:
        df["Cost (£)"] = pd.to_numeric(
            df["gross_concession"].astype(str).str.replace("[£$,]", "", regex=True), errors="coerce"
        ).fillna(0)
    else:
        df["Cost (£)"] = 0.0

    # Assign shift from Exception Open Time
    df["Shift"] = df.apply(assign_shift_pse, axis=1)

    # Standardise Effective column
    df["Effective"] = df["Effective (Y/N)"].str.strip().str.upper()
    df["Is Effective"] = df["Effective"] == "Y"

    # Standardise SLA column
    if "SLA (Y/N)" in df.columns:
        df["SLA Met"] = df["SLA (Y/N)"].str.strip().str.upper() == "Y"
    else:
        df["SLA Met"] = False

    # Clean Problem_Solver — strip @amazon.com for display
    if "Problem_Solver" in df.columns:
        df["PS Display"] = df["Problem_Solver"].astype(str).str.replace("@amazon.com", "", regex=False).str.strip()
    else:
        df["PS Display"] = "Unknown"

    # Standardise Process
    if "Process" in df.columns:
        df["Process"] = df["Process"].str.strip().str.upper()

    # Standardise Category
    if "Category" in df.columns:
        df["Category"] = df["Category"].str.strip()

    return df


def clean_scc(df):
    """Clean SCC data — keep only location columns needed for merge."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Standardise the key merge column
    if "Tracking ID" in df.columns:
        df["Tracking ID"] = df["Tracking ID"].astype(str).str.strip()

    # Keep only what we need from SCC
    keep_cols = ["Tracking ID", "Sort Zone", "Aisle", "Cluster",
                 "Package Length", "Package Width", "Package Height", "DSP Name"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def merge_pse_scc(pse_df, scc_df):
    """Merge PSE data with SCC location data on Scannable ID / Tracking ID."""
    pse = pse_df.copy()
    scc = clean_scc(scc_df.copy())

    # Standardise merge keys
    pse["_merge_key"] = pse["Scannable ID"].astype(str).str.strip()
    scc["_merge_key"] = scc["Tracking ID"].astype(str).str.strip()

    # Left join — PSE is master
    merged = pse.merge(scc, on="_merge_key", how="left", suffixes=("", "_scc"))
    merged = merged.drop(columns=["_merge_key"], errors="ignore")

    return merged


def get_date_range(df):
    """Get human-readable date range from PSE data."""
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if len(dates) > 0:
            s = dates.min().strftime("%d %b %Y")
            e = dates.max().strftime("%d %b %Y")
            return s if s == e else f"{s} – {e}"
    # Fallback to Exception Open DT
    if "Exception Open DT" in df.columns:
        valid = df["Exception Open DT"].dropna()
        if len(valid) > 0:
            s = valid.min().strftime("%d %b %Y")
            e = valid.max().strftime("%d %b %Y")
            return s if s == e else f"{s} – {e}"
    return ""


def fmt_pct(num, denom):
    """Format as percentage string."""
    if denom == 0:
        return "0.0%"
    return f"{round(num / denom * 100, 1)}%"


def fmt_cost(val):
    """Format as £ currency."""
    try:
        if pd.isna(val):
            return "£0.00"
        return f"£{float(val):,.2f}"
    except (ValueError, TypeError):
        return "£0.00"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP — INPUT SECTION
# ═══════════════════════════════════════════════════════════════════════════════

mode = st.radio("Mode:", ["📖 Guide", "Single Station", "Multi-Station / Compare"], horizontal=True, key="mode")

if mode == "📖 Guide":
    st.markdown("### 📖 How to Use This Tool")
    with st.expander("🚀 Quick Start — Read This First", expanded=True):
        st.markdown("""
**What you need:**

| File | Where to get it | What it contains |
|------|----------------|-----------------|
| **PSE Dashboard CSV** | PSE Dashboard → Raw Data → Export CSV | Every problem-solve event (process, category, effectiveness, time, solver) |
| **SCC CSV** (optional but recommended) | SCC → paste Scannable IDs → Export | Where parcels are in the station (cluster, aisle, sort zone) |

**How to use:**
1. Go to **PSE Dashboard** → Raw Data → Export the CSV
2. (Optional) Copy the Scannable IDs from that export into **SCC** → Export
3. Come here → Upload file(s)
4. Use the **filters** to focus on a specific process (Pick, Stow, Induct, Dispatch)
5. Toggle between Effective / Ineffective / All to see different views
6. Look at the **Summary** tab first → identify the biggest problem areas
7. Use **Locations** tab to find which aisles/clusters to walk

**💡 SCC enriches your data with physical location (Cluster, Aisle, Sort Zone).**
Without it, you can still see Route data from PSE, but you won't get the full drill-down.
""")
    with st.expander("📊 What each tab does"):
        st.markdown("""
| Tab | What it helps you do |
|-----|---------------------|
| 📊 **Summary** | Full picture — effectiveness rate, by process, by category, by shift |
| 📍 **Locations** | Worst clusters/aisles/sort zones by ineffective count + rate |
| 👤 **Problem Solvers** | Who's effective, who isn't, resolution times |
| ⏰ **Time & Cycles** | When problems happen (hour, shift, cycle) |
| 💰 **Cost & DEA** | Financial impact — concessions, DEA misses |
| 🔬 **Analysis & Trend** | Statistical findings + week-over-week tracking |
| 💾 **Export** | Download filtered clean data |
""")
    with st.expander("🎯 How Recommendations Work"):
        st.markdown("""
The tool cross-references **location × category × effectiveness** to tell you:

- **Where** to focus (which cluster/aisle has the most problems)
- **What** the problem is (which category dominates in that location)
- **Who** is solving there (and their effectiveness rate)
- **When** it happens (which shift/cycle)

Example output:
> 🔴 **Priority 1:** Aisle CA_A197 — 12 STOW events, 8 ineffective (33% eff. rate)
> Root cause: STOW_WRONG_AISLE_CLUSTER (75% of issues here)
> → Walk this aisle. Check bin labels and stow guidance.
""")

elif mode == "Single Station":
    st.caption("Upload your PSE Dashboard CSV. Optionally add SCC for cluster/aisle/sort zone drill-down.")

    # ─── FILE UPLOAD ──────────────────────────────────────────────────────────
    c_pse, c_scc = st.columns(2)
    with c_pse:
        pse_file = st.file_uploader("🔧 PSE Dashboard CSV", type="csv", key="pse")
    with c_scc:
        scc_file = st.file_uploader("📋 SCC CSV (optional — adds location)", type="csv", key="scc")

    if pse_file:
        # ─── READ & VALIDATE PSE ─────────────────────────────────────────────
        try:
            pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
        except Exception as e:
            st.error(f"❌ Error reading PSE CSV: {e}")
            st.stop()

        pse_miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
        if pse_miss:
            st.error(f"❌ PSE CSV missing required columns: {pse_miss}")
            st.stop()

        # ─── READ & VALIDATE SCC (if provided) ───────────────────────────────
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
                    st.warning(f"⚠️ SCC CSV missing columns: {scc_miss} — location drill-down may be limited.")

        # ─── CLEAN & MERGE ────────────────────────────────────────────────────
        df = clean_pse(pse_df)

        if scc_df is not None:
            df = merge_pse_scc(df, scc_df)
            matched = df["Cluster"].notna().sum() if "Cluster" in df.columns else 0
            st.success(f"✅ **{len(df)} PSE events** loaded — SCC matched: {matched}/{len(df)}")
        else:
            # Ensure location columns exist even without SCC (use Route as proxy)
            for col in ["Cluster", "Aisle", "Sort Zone"]:
                if col not in df.columns:
                    df[col] = None
            st.success(f"✅ **{len(df)} PSE events** loaded (no SCC — using Route data only)")

        # ─── FILTERS (Process + Effectiveness + Category) ────────────────────
        st.markdown("---")
        st.markdown("### 🎛️ Filters")

        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            available_processes = sorted(df["Process"].dropna().unique().tolist())
            selected_processes = st.multiselect(
                "Process:",
                options=available_processes,
                default=available_processes,
                key="filter_process"
            )

        with f_col2:
            eff_filter = st.radio(
                "Effectiveness:",
                ["All", "Ineffective Only", "Effective Only"],
                horizontal=True,
                key="filter_eff"
            )

        with f_col3:
            available_categories = sorted(df["Category"].dropna().unique().tolist())
            selected_categories = st.multiselect(
                "Category:",
                options=available_categories,
                default=available_categories,
                key="filter_cat"
            )

        # ─── APPLY FILTERS ───────────────────────────────────────────────────
        filtered = df.copy()

        # Process filter
        if selected_processes:
            filtered = filtered[filtered["Process"].isin(selected_processes)]
        else:
            filtered = filtered[filtered["Process"].isin([])]  # empty if nothing selected

        # Effectiveness filter
        if eff_filter == "Ineffective Only":
            filtered = filtered[filtered["Is Effective"] == False]
        elif eff_filter == "Effective Only":
            filtered = filtered[filtered["Is Effective"] == True]

        # Category filter
        if selected_categories:
            filtered = filtered[filtered["Category"].isin(selected_categories)]
        else:
            filtered = filtered[filtered["Category"].isin([])]

        total = len(filtered)
        total_unfiltered = len(df)

        if total == 0:
            st.warning("No events match your filters. Try broadening your selection.")
            st.stop()

        # ─── HEADLINE METRICS ────────────────────────────────────────────────
        dr = get_date_range(filtered)
        eff_count = int(filtered["Is Effective"].sum())
        ineff_count = total - eff_count
        sla_count = int(filtered["SLA Met"].sum())
        total_cost = filtered["Cost (£)"].sum()
        avg_res = filtered["Resolution Min"].dropna()

        st.markdown("---")
        if dr:
            st.caption(f"📅 Date range: **{dr}** | Showing {total} of {total_unfiltered} events")
        else:
            st.caption(f"Showing {total} of {total_unfiltered} events")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Events", total)
        c2.metric("Effective", f"{eff_count} ({fmt_pct(eff_count, total)})")
        c3.metric("Ineffective", f"{ineff_count} ({fmt_pct(ineff_count, total)})")
        c4.metric("SLA Met", fmt_pct(sla_count, total))
        c5.metric("Concession Cost", fmt_cost(total_cost))
        c6.metric("Avg Resolution", f"{avg_res.mean():.0f} min" if len(avg_res) > 0 else "N/A")

        # ─── TABS (placeholders — built in subsequent parts) ─────────────────
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7 = st.tabs([
            "📊 Summary", "📍 Locations", "👤 Problem Solvers",
            "⏰ Time & Cycles", "💰 Cost & DEA", "🔬 Analysis & Trend", "💾 Export"
        ])

        with t1:
            st.info("📊 Summary tab — coming next.")
        with t2:
            st.info("📍 Locations tab — coming next.")
        with t3:
            st.info("👤 Problem Solvers tab — coming next.")
        with t4:
            st.info("⏰ Time & Cycles tab — coming next.")
        with t5:
            st.info("💰 Cost & DEA tab — coming next.")
        with t6:
            st.info("🔬 Analysis & Trend tab — coming next.")
        with t7:
            st.info("💾 Export tab — coming next.")

    else:
        st.info("👆 Upload your PSE Dashboard CSV to get started.")

elif mode == "Multi-Station / Compare":
    st.info("🏢 Multi-Station mode — coming in a later build.")
