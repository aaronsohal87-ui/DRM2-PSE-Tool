import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

st.set_page_config(page_title="DRM2 PSE Heatmap", page_icon="🔧", layout="wide")
st.title("🔧 DRM2 PSE Heatmap")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
PROCESS_ORDER = ["INDUCT", "STOW", "PICK", "DISPATCH"]
PROCESS_COLORS = {"INDUCT": "midnightblue", "STOW": "darkorange", "PICK": "darkgreen", "DISPATCH": "firebrick"}

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
    return df.groupby("PS Display")["Shift"].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
    ).to_dict()

def fmt_pct(num, denom):
    if denom == 0: return "0%"
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
            s, e = dates.min().strftime("%d %b %Y"), dates.max().strftime("%d %b %Y")
            return s if s == e else f"{s} – {e}"
    return ""

def make_bar_horiz(data, title, color="steelblue", max_bars=10):
    data = data.head(max_bars)
    if len(data) == 0: return plt.subplots(figsize=(7, 2))[0]
    h = max(2, len(data)*0.3)
    fig, ax = plt.subplots(figsize=(7, h))
    labs = trunc(data.index)
    ax.barh(labs, data.values, color=color); ax.invert_yaxis()
    mx = data.values.max() if len(data) > 0 else 1
    ax.set_xlim(right=mx * 1.18)
    for i, v in enumerate(data.values):
        ax.text(v + mx*0.02, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Count", fontsize=8); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    mx = max([data[s] for s in SHIFT_ORDER]) if any(data[s] > 0 for s in SHIFT_ORDER) else 1
    ax.set_ylim(top=mx * 1.25)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift", fontsize=8); ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_eff_bar(df, group_col, title, top_n=10):
    grouped = df.groupby(group_col).agg(
        Total=("Scannable ID", "count"), Effective=("Is Effective", "sum")
    ).sort_values("Total", ascending=False).head(top_n)
    grouped["Ineffective"] = grouped["Total"] - grouped["Effective"]
    grouped["Eff %"] = (grouped["Effective"] / grouped["Total"] * 100).round(1)
    if len(grouped) == 0:
        fig, ax = plt.subplots(figsize=(7, 2)); ax.text(0.5, 0.5, "No data", ha="center"); return fig
    h = max(2, len(grouped)*0.35)
    fig, ax = plt.subplots(figsize=(7, h))
    labs = trunc(grouped.index)
    ax.barh(labs, grouped["Effective"].values, color="#2ecc71", label="Effective")
    ax.barh(labs, grouped["Ineffective"].values, left=grouped["Effective"].values, color="#e74c3c", label="Ineffective")
    ax.invert_yaxis()
    mx = grouped["Total"].max()
    ax.set_xlim(right=mx * 1.25)
    for i, (tot, rate) in enumerate(zip(grouped["Total"].values, grouped["Eff %"].values)):
        ax.text(tot + mx*0.02, i, f"{int(tot)} ({rate}%)", va="center", fontsize=7)
    ax.set_xlabel("Events", fontsize=8); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7); ax.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_pse(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in ["Exception Open Time", "Resolution Time", "PSS Event Time", "Shipment Status Datetime"]:
        if col in df.columns:
            df[col + "_DT"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if "Exception Open Time_DT" in df.columns: df["Exception Open DT"] = df["Exception Open Time_DT"]
    if "PSS Event Time_DT" in df.columns: df["PSS Event DT"] = df["PSS Event Time_DT"]
    if "gross_concession" in df.columns:
        df["Cost (£)"] = pd.to_numeric(df["gross_concession"].astype(str).str.replace("[£$,]","",regex=True), errors="coerce").fillna(0)
    else:
        df["Cost (£)"] = 0.0
    df["Shift"] = df.apply(assign_shift_pse, axis=1)
    df["Effective"] = df["Effective (Y/N)"].astype(str).str.strip().str.upper()
    df["Is Effective"] = df["Effective"] == "Y"
    if "SLA (Y/N)" in df.columns:
        df["SLA Met"] = df["SLA (Y/N)"].astype(str).str.strip().str.upper() == "Y"
    else:
        df["SLA Met"] = False
    if "Problem_Solver" in df.columns:
        df["PS Display"] = df["Problem_Solver"].astype(str).str.replace("@amazon.com","",regex=False).str.strip()
    else:
        df["PS Display"] = "Unknown"
    if "Process" in df.columns: df["Process"] = df["Process"].astype(str).str.strip().str.upper()
    if "Category" in df.columns: df["Category"] = df["Category"].astype(str).str.strip()
    if "dea_miss" in df.columns:
        df["DEA Miss"] = pd.to_numeric(df["dea_miss"], errors="coerce").fillna(0).astype(int)
    else:
        df["DEA Miss"] = 0
    return df

def clean_scc(df):
    df = df.copy(); df.columns = df.columns.str.strip()
    if "Tracking ID" in df.columns: df["Tracking ID"] = df["Tracking ID"].astype(str).str.strip()
    keep = [c for c in ["Tracking ID","Sort Zone","Aisle","Cluster"] if c in df.columns]
    return df[keep]

def merge_pse_scc(pse_df, scc_df):
    pse = pse_df.copy(); scc = clean_scc(scc_df.copy())
    pse["_mk"] = pse["Scannable ID"].astype(str).str.strip()
    scc["_mk"] = scc["Tracking ID"].astype(str).str.strip()
    merged = pse.merge(scc, on="_mk", how="left", suffixes=("","_scc"))
    return merged.drop(columns=["_mk"], errors="ignore")

def filter_uk_ids(df):
    mask = df["Scannable ID"].astype(str).str.strip().str.startswith("UK")
    return df[mask].copy(), (~mask).sum()

def compute_health_score(df, total):
    if total == 0: return 5, "🟡", "No data", []
    score = 10; reasons = []
    eff = df["Is Effective"].sum() / total
    if eff < 0.5: score -= 3; reasons.append(f"Eff {eff*100:.0f}%")
    elif eff < 0.65: score -= 2; reasons.append(f"Eff {eff*100:.0f}%")
    elif eff < 0.75: score -= 1; reasons.append(f"Eff {eff*100:.0f}%")
    sla = df["SLA Met"].sum() / total
    if sla < 0.5: score -= 2; reasons.append(f"SLA {sla*100:.0f}%")
    elif sla < 0.7: score -= 1; reasons.append(f"SLA {sla*100:.0f}%")
    dea = df["DEA Miss"].sum()
    if dea >= 5: score -= 2; reasons.append(f"{int(dea)} DEA misses")
    elif dea >= 2: score -= 1; reasons.append(f"{int(dea)} DEA miss(es)")
    score = max(1, min(10, score))
    if score >= 8: return score, "🟢", "Good", reasons
    elif score >= 5: return score, "🟡", "Needs attention", reasons
    return score, "🔴", "Action required", reasons


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def render_summary_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("📋 Tracking IDs — Copy into SCC", expanded=False):
        st.markdown("""
**To get location data (cluster, aisle, sort zone):**
1. Select all the IDs below (click in box → Ctrl+A)
2. Copy (Ctrl+C)
3. Paste into SCC search → Export as CSV
4. Upload that CSV here as the SCC file
""")
        uk_ids = sorted(df[df["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].astype(str).str.strip().unique())
        if len(uk_ids) > 0:
            st.caption(f"{len(uk_ids)} unique UK IDs")
            st.code("\n".join(uk_ids), language=None)
        else:
            st.warning("No UK IDs.")

    with st.expander("📦 By Process", expanded=True):
        proc = df.groupby("Process").agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum"), SLA=("SLA Met","sum")
        ).reindex(PROCESS_ORDER, fill_value=0)
        proc["Ineffective"] = proc["Total"] - proc["Effective"]
        proc["Eff %"] = (proc["Effective"]/proc["Total"]*100).round(1)
        proc["SLA %"] = (proc["SLA"]/proc["Total"]*100).round(1)
        c1, c2 = st.columns([1,1])
        with c1: st.dataframe(proc[["Total","Effective","Ineffective","Eff %","SLA %"]], use_container_width=True)
        with c2:
            pt = proc["Total"]; pt = pt[pt>0]
            if len(pt) > 0:
                fig, ax = plt.subplots(figsize=(3,2.5))
                ax.pie(pt.values, labels=pt.index, colors=[PROCESS_COLORS.get(p,"gray") for p in pt.index],
                       autopct="%1.0f%%", startangle=90, textprops={"fontsize":7})
                ax.set_title(f"By Process ({dr})", fontsize=8); plt.tight_layout(); st.pyplot(fig)

    with st.expander("🏷️ By Category", expanded=True):
        st.pyplot(make_eff_bar(df, "Category", f"Categories ({dr})"))

    with st.expander("🕐 Hour of Day"):
        if "Exception Open DT" in df.columns:
            hours = df["Exception Open DT"].dropna().dt.hour
            if len(hours) > 0:
                hc = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig, ax = plt.subplots(figsize=(8,2.5))
                ax.bar(range(24), hc.values, color=[SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h,""),"gray") for h in range(24)])
                for h in range(24):
                    if hc.values[h]>0: ax.text(h, hc.values[h]+0.1, str(int(hc.values[h])), ha="center", fontsize=6)
                ax.set_xlabel("Hour", fontsize=8); ax.set_ylabel("Events", fontsize=8)
                ax.set_xticks(range(24)); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
                st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: LOCATIONS (Cluster drill-down focused)
# ═══════════════════════════════════════════════════════════════════════════════

def render_locations_tab(df, total, dr):
    if total == 0: st.warning("No data."); return
    has_scc = "Cluster" in df.columns and df["Cluster"].notna().any()

    if not has_scc:
        st.warning("⚠️ No SCC data. Go to **Summary** tab → copy tracking IDs → paste into SCC → export CSV → upload here.")
        return

    # Overview: top clusters chart
    with st.expander("📍 Cluster Overview (top 10)", expanded=True):
        st.pyplot(make_eff_bar(df[df["Cluster"].notna()], "Cluster", f"Top 10 Clusters ({dr})", top_n=10))

    # Main view: drill into a cluster
    st.markdown("### 🔍 Cluster Deep Dive")
    st.caption("Select a cluster to see everything inside it — aisles, categories, which shift, who's solving there.")
    clusters = df["Cluster"].dropna().value_counts()
    cluster_list = clusters.index.tolist()

    if cluster_list:
        sel = st.selectbox("Select cluster:", cluster_list, format_func=lambda x: f"{x} ({int(clusters[x])} events)", key="drill_cl")
        filt = df[df["Cluster"] == sel]
        eff_n = int(filt["Is Effective"].sum())
        ineff_n = len(filt) - eff_n

        c1, c2, c3 = st.columns(3)
        c1.metric("Events", len(filt))
        c2.metric("Effective", f"{eff_n} ({fmt_pct(eff_n, len(filt))})")
        c3.metric("Ineffective", f"{ineff_n} ({fmt_pct(ineff_n, len(filt))})")

        # Aisles table (NOT graph — too many aisles)
        st.markdown("**Aisles in this cluster:**")
        if "Aisle" in filt.columns and filt["Aisle"].notna().any():
            aisle_data = filt.groupby("Aisle").agg(
                Events=("Scannable ID","count"), Effective=("Is Effective","sum")
            ).sort_values("Events", ascending=False)
            aisle_data["Ineffective"] = aisle_data["Events"] - aisle_data["Effective"]
            aisle_data["Eff %"] = (aisle_data["Effective"]/aisle_data["Events"]*100).round(1)
            aisle_data.index.name = "Aisle"
            st.dataframe(aisle_data[["Events","Effective","Ineffective","Eff %"]], use_container_width=True)
        else:
            st.info("No aisle data for this cluster.")

        # Categories in this cluster
        st.markdown("**What's going wrong here:**")
        cat_data = filt.groupby("Category").agg(
            Events=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).sort_values("Events", ascending=False)
        cat_data["Ineffective"] = cat_data["Events"] - cat_data["Effective"]
        cat_data["Eff %"] = (cat_data["Effective"]/cat_data["Events"]*100).round(1)
        st.dataframe(cat_data[["Events","Effective","Ineffective","Eff %"]], use_container_width=True)

        # Which shift
        st.markdown("**Which shift:**")
        shift_data = filt[filt["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Events=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        shift_data["Eff %"] = (shift_data["Effective"]/shift_data["Events"]*100).round(1)
        shift_data["Window"] = [SHIFT_DEFINITIONS.get(s,"") for s in shift_data.index]
        st.dataframe(shift_data[["Events","Effective","Eff %","Window"]], use_container_width=True)

        # Who is solving in this cluster
        st.markdown("**Who is problem-solving here:**")
        ps_in_cluster = filt.groupby("PS Display").agg(
            Events=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).sort_values("Events", ascending=False)
        ps_in_cluster["Ineffective"] = ps_in_cluster["Events"] - ps_in_cluster["Effective"]
        ps_in_cluster["Eff %"] = (ps_in_cluster["Effective"]/ps_in_cluster["Events"]*100).round(1)
        st.dataframe(ps_in_cluster[["Events","Effective","Ineffective","Eff %"]].head(10), use_container_width=True)

        # Sort Zone (if available)
        if "Sort Zone" in filt.columns and filt["Sort Zone"].notna().any():
            st.markdown("**Sort Zones:**")
            sz = filt["Sort Zone"].dropna().value_counts()
            tbl = sz.reset_index(); tbl.columns = ["Sort Zone","Events"]; tbl.index = range(1,len(tbl)+1)
            st.dataframe(tbl, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PROBLEM SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_ps_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    # DATA SOURCE NOTICE
    st.markdown("""
> ⚠️ **Data accuracy notice**
>
> This data comes directly from the **PSE Dashboard CSV export**.
> - **Associate name** = `Problem_Solver` column in the CSV
> - **Eff %** = events marked `Y` in `Effective (Y/N)` column ÷ total events × 100
> - **SLA %** = events marked `Y` in `SLA (Y/N)` column ÷ total events × 100
> - **Shift** = determined by the hour of `Exception Open Time` (when the PS event was created)
>
> **Before taking action:** Verify with the associate. System errors, misattributed scans,
> or shared logins can cause incorrect data. This is a starting point for investigation, not proof.
""")

    solver_shifts = get_solver_shift(df)

    ps = df.groupby("PS Display").agg(
        Total=("Scannable ID","count"),
        Effective=("Is Effective","sum"),
        SLA=("SLA Met","sum"),
        Cost=("Cost (£)","sum")
    )
    ps["Ineffective"] = ps["Total"] - ps["Effective"]
    ps["Eff %"] = (ps["Effective"]/ps["Total"]*100).round(1)
    ps["SLA %"] = (ps["SLA"]/ps["Total"]*100).round(1)
    ps["Shift"] = ps.index.map(lambda x: solver_shifts.get(x, "?"))

    # Sort: 0% effective first (never resolved anything), then ascending by Eff%
    ranked = ps[ps["Total"] >= 3].copy()
    # Put 0% at top, then sort rest ascending
    ranked["_sort"] = ranked["Eff %"]
    ranked = ranked.sort_values("_sort", ascending=True).drop(columns=["_sort"])

    with st.expander("👤 All Associates — Ranked (3+ events)", expanded=True):
        st.caption("Associates with **0% effectiveness** (never resolved a single package) are at the top. Then sorted worst → best.")
        if len(ranked) > 0:
            avg_eff = ranked["Eff %"].mean()
            avg_sla = ranked["SLA %"].mean()
            st.markdown(f"**Team average: {avg_eff:.0f}% effective | {avg_sla:.0f}% SLA**")

            # Highlight the 0% people
            zero_eff = ranked[ranked["Eff %"] == 0]
            if len(zero_eff) > 0:
                st.error(f"🚨 **{len(zero_eff)} associate(s) with 0% effectiveness** — never resolved a single package:")
                for name, row in zero_eff.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']} shift] — {int(row['Total'])} events, 0 effective")

            display = ranked[["Shift","Total","Effective","Ineffective","Eff %","SLA %"]].reset_index()
            display = display.rename(columns={"PS Display":"Associate"})
            display.index = range(1, len(display)+1)
            display.index.name = "Rank"
            st.dataframe(display, use_container_width=True, height=min(700, 35*len(display)+40))

    with st.expander("🔴 Flagged — Below Average"):
        if len(ranked) >= 3:
            avg_eff = ranked["Eff %"].mean()
            avg_sla = ranked["SLA %"].mean()

            st.markdown("**Below average effectiveness:**")
            flagged_eff = ranked[ranked["Eff %"] < avg_eff - 10]
            if len(flagged_eff) > 0:
                for name, row in flagged_eff.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']}]: **{row['Eff %']}%** eff ({int(row['Total'])} events)")
            else:
                st.success(f"✅ Nobody >10pp below average ({avg_eff:.0f}%).")

            st.markdown("**Below average SLA:**")
            flagged_sla = ranked[ranked["SLA %"] < avg_sla - 10]
            if len(flagged_sla) > 0:
                for name, row in flagged_sla.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']}]: **{row['SLA %']}%** SLA ({int(row['Total'])} events)")
            else:
                st.success(f"✅ Nobody >10pp below SLA average ({avg_sla:.0f}%).")

    with st.expander("🎯 Where are they failing? (Category breakdown)"):
        st.caption("Which categories each associate is worst at. Only combos with 3+ events shown.")
        ps_cat = df.groupby(["PS Display","Category"]).agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).reset_index()
        ps_cat["Eff %"] = (ps_cat["Effective"]/ps_cat["Total"]*100).round(1)
        worst = ps_cat[ps_cat["Total"]>=3].sort_values("Eff %", ascending=True).head(20)
        if len(worst) > 0:
            worst["Ineffective"] = worst["Total"] - worst["Effective"]
            out = worst[["PS Display","Category","Total","Ineffective","Eff %"]].rename(columns={"PS Display":"Associate"})
            out.index = range(1, len(out)+1)
            st.dataframe(out, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_cycles_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("🔄 By Actual Cycle", expanded=True):
        if "Actual Cycle" in df.columns:
            cyc = df.groupby("Actual Cycle").agg(
                Total=("Scannable ID","count"), Effective=("Is Effective","sum"), SLA=("SLA Met","sum")
            ).sort_values("Total", ascending=False)
            cyc["Ineffective"] = cyc["Total"] - cyc["Effective"]
            cyc["Eff %"] = (cyc["Effective"]/cyc["Total"]*100).round(1)
            cyc["SLA %"] = (cyc["SLA"]/cyc["Total"]*100).round(1)
            st.dataframe(cyc[["Total","Effective","Ineffective","Eff %","SLA %"]], use_container_width=True)

    with st.expander("📅 By Planned Cycle"):
        if "Planned Cycle" in df.columns:
            pc = df.groupby("Planned Cycle").agg(
                Total=("Scannable ID","count"), Effective=("Is Effective","sum")
            ).sort_values("Total", ascending=False)
            pc["Eff %"] = (pc["Effective"]/pc["Total"]*100).round(1)
            st.dataframe(pc[["Total","Effective","Eff %"]], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: COST & DEA
# ═══════════════════════════════════════════════════════════════════════════════

def render_cost_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    tc = df["Cost (£)"].sum()
    with_cost = (df["Cost (£)"]>0).sum()
    dea = int(df["DEA Miss"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Concession Cost", fmt_cost(tc))
    c2.metric("Events with Concession", with_cost)
    c3.metric("DEA Misses", dea)

    with st.expander("💰 What is concession cost?", expanded=False):
        st.markdown("""
**Concession cost** = money Amazon refunded to the customer because of this issue.

This comes from the `gross_concession` column in the PSE export. If a package was
damaged, lost, or delayed because of a PS failure, the customer may get a refund —
that's the concession.

- **"Cost (Ineffective)"** = total concessions on events where the associate marked the PS as NOT effective
- **"Cost (Effective)"** = total concessions on events where PS WAS effective (issue still cost money but was handled correctly)
""")

    with st.expander("💰 Cost by Category", expanded=True):
        cdf = df[df["Cost (£)"]>0]
        if len(cdf) > 0:
            cc = cdf.groupby("Category").agg(Events=("Scannable ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            cc["Avg"] = (cc["Cost"]/cc["Events"]).apply(fmt_cost); cc["Cost"] = cc["Cost"].apply(fmt_cost)
            cc.index = range(1,len(cc)+1); st.dataframe(cc, use_container_width=True)
        else:
            st.info("No concession costs in this data.")

    with st.expander("💰 Cost: Effective vs Ineffective PS"):
        ineff_cost = df[~df["Is Effective"]]["Cost (£)"].sum()
        eff_cost = df[df["Is Effective"]]["Cost (£)"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Cost (PS was effective)", fmt_cost(eff_cost))
        c2.metric("Cost (PS was ineffective)", fmt_cost(ineff_cost))
        if ineff_cost > 0:
            st.error(f"💡 **{fmt_cost(ineff_cost)}** in customer refunds where PS failed. Better PS could have prevented some of this.")

    # DEA MISSES — organised by shift with drill-down
    with st.expander("🎯 DEA Misses — By Shift", expanded=True):
        st.caption("DEA Miss = package wasn't dispatched when it should have been. Organised by shift so you know when it happened.")
        dea_events = df[df["DEA Miss"]>0]
        if len(dea_events) > 0:
            st.error(f"🚨 {len(dea_events)} event(s) with DEA misses")

            # By shift summary
            dea_by_shift = dea_events[dea_events["Shift"].isin(SHIFT_ORDER)].groupby("Shift").size().reindex(SHIFT_ORDER, fill_value=0)
            st.pyplot(make_bar_shift(dea_by_shift, "DEA Misses by Shift"))

            # Drill down per shift
            for s in SHIFT_ORDER:
                s_events = dea_events[dea_events["Shift"]==s]
                if len(s_events) > 0:
                    with st.expander(f"{s} shift — {len(s_events)} DEA miss(es) ({SHIFT_DEFINITIONS[s]})"):
                        cols = [c for c in ["Scannable ID","Process","Category","PS Display","dea_bucket","Status"] if c in s_events.columns]
                        st.dataframe(s_events[cols].reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No DEA misses.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HOLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_holes_tab(df, total, dr):
    if total == 0: st.warning("No data."); return
    st.markdown("### 🕳️ Holes in Problem Solve")
    st.caption("Packages that needed problem-solving more than once. Organised by what happened — did it get fixed or not?")

    # Find repeat packages
    id_counts = df.groupby("Scannable ID").size()
    repeats = id_counts[id_counts > 1].sort_values(ascending=False)

    if len(repeats) == 0:
        st.success("✅ No packages needed problem-solving more than once.")
        return

    st.error(f"🚨 **{len(repeats)} packages** were problem-solved more than once.")

    # Build comparison data with pattern classification
    repeat_ids = repeats.index.tolist()
    rdf = df[df["Scannable ID"].isin(repeat_ids)].sort_values(["Scannable ID", "Exception Open DT"])

    comparisons = []
    for tid in repeat_ids:
        events = rdf[rdf["Scannable ID"]==tid].reset_index(drop=True)
        effs = events["Effective"].tolist()
        pattern = " then ".join(["Effective" if e=="Y" else "Ineffective" for e in effs[:2]])
        same_solver = len(set(events["PS Display"].tolist()[:2])) == 1

        row = {
            "Tracking ID": tid,
            "Pattern": pattern,
            "Same Associate?": "Yes" if same_solver else "No",
            "1st Process": events.iloc[0].get("Process","") if len(events)>0 else "",
            "1st Category": events.iloc[0].get("Category","") if len(events)>0 else "",
            "1st Associate": events.iloc[0].get("PS Display","") if len(events)>0 else "",
            "1st Result": "Effective" if events.iloc[0].get("Effective","")=="Y" else "Ineffective",
            "2nd Process": events.iloc[1].get("Process","") if len(events)>1 else "",
            "2nd Category": events.iloc[1].get("Category","") if len(events)>1 else "",
            "2nd Associate": events.iloc[1].get("PS Display","") if len(events)>1 else "",
            "2nd Result": "Effective" if events.iloc[1].get("Effective","")=="Y" else "Ineffective" if len(events)>1 else "",
        }
        comparisons.append(row)

    comp_df = pd.DataFrame(comparisons)

    # Pattern summary
    pattern_counts = comp_df["Pattern"].value_counts()
    with st.expander("📊 Pattern Summary", expanded=True):
        st.markdown("""
| Pattern | What it means | Concern Level |
|---------|--------------|---------------|
| **Ineffective then Ineffective** | Failed BOTH times — nobody could fix it | 🔴 Critical |
| **Effective then Effective** | Marked as fixed twice — why did it come back? | 🟠 Suspicious |
| **Effective then Ineffective** | Fixed once, new problem appeared or original fix failed | 🟡 Investigate |
| **Ineffective then Effective** | Failed first, fixed second — normal recovery | 🟢 OK |
""")
        for pattern, count in pattern_counts.items():
            same_solver_n = comp_df[(comp_df["Pattern"]==pattern) & (comp_df["Same Associate?"]=="Yes")].shape[0]
            st.markdown(f"- **{pattern}**: {count} packages ({same_solver_n} by same associate both times)")

    # Show each pattern group
    pattern_labels = {
        "Ineffective then Ineffective": ("🔴 Never Fixed", "Failed both times. Nobody resolved this package. Needs escalation."),
        "Effective then Effective": ("🟠 Came Back After 'Fix'", "Marked effective but the package returned. Was the first fix real?"),
        "Effective then Ineffective": ("🟡 New Problem After Fix", "First issue fixed, but a new issue appeared (or the fix didn't hold)."),
        "Ineffective then Effective": ("🟢 Eventually Fixed", "Failed first attempt, someone fixed it second time. Normal recovery."),
    }

    for pattern in ["Ineffective then Ineffective", "Effective then Effective", "Effective then Ineffective", "Ineffective then Effective"]:
        subset = comp_df[comp_df["Pattern"]==pattern]
        if len(subset) == 0: continue
        title, desc = pattern_labels.get(pattern, (pattern, ""))
        with st.expander(f"{title} ({len(subset)} packages)"):
            st.caption(desc)
            display = subset[[
                "Tracking ID", "Same Associate?",
                "1st Process", "1st Category", "1st Associate", "1st Result",
                "2nd Process", "2nd Category", "2nd Associate", "2nd Result"
            ]].reset_index(drop=True)
            display.index = range(1, len(display)+1)
            st.dataframe(display, use_container_width=True)

    # Worst outcomes — failed PS with cost
    with st.expander("📉 Costly Failures — Ineffective PS + Customer Refund"):
        st.caption("""
Events where PS was marked Ineffective AND the customer received a refund (concession).
Data source: `Effective (Y/N) = N` AND `gross_concession > 0` in PSE export.
""")
        bad = df[(~df["Is Effective"]) & (df["Cost (£)"]>0)].sort_values("Cost (£)", ascending=False)
        if len(bad) > 0:
            st.error(f"{len(bad)} events — {fmt_cost(bad['Cost (£)'].sum())} total concessions")
            cols = [c for c in ["Scannable ID","Process","Category","PS Display","Shift","Cost (£)"] if c in bad.columns]
            st.dataframe(bad[cols].head(20).reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No costly failures.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: TREND
# ═══════════════════════════════════════════════════════════════════════════════

def render_trend_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    st.markdown("### 📈 Week-over-Week Trend")
    st.markdown("""
**How this works:**

You track your PSE numbers over multiple weeks to see if things are getting better or worse.

**Option 1 — Type values:** Open PSE Dashboard each week, note down the total events and
how many were effective, and type them in below.

**Option 2 — Upload CSVs:** Export a PSE Dashboard CSV each week (one file per week).
Upload them here and the tool calculates everything automatically.

You need **at least 2 weeks** to see a trend. 4+ weeks is ideal.
""")
    st.markdown("---")

    tm = st.selectbox("How do you want to input your weekly data?", ["📝 I'll type the numbers", "📂 I'll upload a CSV per week"], key="tm")

    if tm == "📝 I'll type the numbers":
        st.caption("Open PSE Dashboard for each week → note Total events and Effective count → type below.")
        nw = st.slider("How many weeks of data do you have?", 2, 12, 4, key="tw_n")
        weeks = []
        for i in range(nw):
            with st.expander(f"Week {i+1}", expanded=(i<3)):
                wl = st.text_input("Week label (e.g. W28, 4 Aug):", value=f"W{i+1}", key=f"tw_l{i}")
                wt = st.number_input("Total PS events that week:", min_value=0, value=0, step=1, key=f"tw_t{i}")
                we = st.number_input("How many were Effective:", min_value=0, value=0, step=1, key=f"tw_e{i}")
                if wt > 0: weeks.append({"Week":wl,"Total":int(wt),"Effective":int(we)})
        _render_trend(weeks)
    else:
        st.caption("Export PSE Dashboard → Raw Data → CSV once per week. Upload one file per week below.")
        nf = st.slider("How many weeks?", 2, 12, 4, key="tf_n")
        weeks = []
        for i in range(nf):
            c1, c2 = st.columns([1,3])
            with c1: wl = st.text_input("Label:", value=f"W{i+1}", key=f"tf_l{i}")
            with c2: fu = st.file_uploader(f"Week {i+1} PSE CSV:", type="csv", key=f"tf_f{i}")
            if fu:
                try:
                    wdf = pd.read_csv(fu, encoding="utf-8-sig")
                    we = int((wdf.get("Effective (Y/N)", pd.Series(dtype=str)).astype(str).str.strip().str.upper()=="Y").sum()) if "Effective (Y/N)" in wdf.columns else 0
                    weeks.append({"Week":wl,"Total":len(wdf),"Effective":we})
                    st.caption(f"✓ {wl}: {len(wdf)} events, {we} effective ({fmt_pct(we,len(wdf))})")
                except Exception as e: st.error(f"Error reading file: {e}")
        _render_trend(weeks)

def _render_trend(weeks):
    if len(weeks) >= 2:
        w = pd.DataFrame(weeks)
        w["Ineffective"] = w["Total"] - w["Effective"]
        w["Eff %"] = (w["Effective"]/w["Total"]*100).round(1)

        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot(w["Week"], w["Total"], marker="o", color="steelblue", linewidth=2, label="Total")
        ax.plot(w["Week"], w["Ineffective"], marker="s", color="#e74c3c", linewidth=1.5, label="Ineffective")
        for _,r in w.iterrows():
            ax.annotate(str(int(r["Total"])), xy=(r["Week"],r["Total"]), xytext=(0,8), textcoords="offset points", ha="center", fontsize=7, color="steelblue")
        ax.set_xlabel("Week"); ax.set_ylabel("Events"); ax.set_title("Weekly Trend", fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(7,2.5))
        ax2.plot(w["Week"], w["Eff %"], marker="o", color="darkgreen", linewidth=2)
        for _,r in w.iterrows():
            ax2.annotate(f"{r['Eff %']}%", xy=(r["Week"],r["Eff %"]), xytext=(0,8), textcoords="offset points", ha="center", fontsize=7)
        ax2.axhline(y=w["Eff %"].mean(), color="gray", linestyle="--", linewidth=1)
        ax2.set_xlabel("Week"); ax2.set_ylabel("Eff %"); ax2.set_title("Effectiveness Trend", fontsize=9)
        ax2.tick_params(labelsize=7); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig2)

        f, l = w.iloc[0]["Eff %"], w.iloc[-1]["Eff %"]
        if l > f+5: st.success(f"📈 Improving: {f}% → {l}%")
        elif l < f-5: st.error(f"📉 Worsening: {f}% → {l}%")
        else: st.info(f"➡️ Stable: {f}% → {l}%")
        st.dataframe(w, use_container_width=True)
    elif len(weeks)==1: st.info("Need at least 2 weeks to show a trend.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def render_export_tab(df, total, dr):
    st.markdown("#### 💾 Export")
    exclude = [c for c in df.columns if c.endswith("_DT") or c=="_mk"]
    clean = [c for c in df.columns if c not in exclude]
    st.download_button("⬇️ Download all data (CSV)", df[clean].to_csv(index=False), "PSE_Data.csv", "text/csv", key="dl")


# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def render_guide():
    st.markdown("### 📖 How to Use")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**Step 1:** Go to PSE Dashboard → Raw Data → Export as CSV

**Step 2:** Upload it here

**Step 3:** (For location data) Go to **Summary** tab → copy tracking IDs → paste into SCC → export SCC CSV → upload here

**Then explore the tabs.**
""")
    with st.expander("📊 What each tab shows"):
        st.markdown("""
| Tab | What you'll find |
|-----|-----------------|
| 📊 **Summary** | Overview by process, category, hour + IDs to copy for SCC |
| 📍 **Locations** | Pick a cluster → see aisles, categories, shifts, who's solving there |
| 👤 **Associates** | Ranked worst→best + who has 0% effectiveness + who's failing SLA |
| 🔄 **Cycles** | Breakdown by dispatch cycle (CYCLE_1, HV_A, etc.) |
| 💰 **Cost & DEA** | Money lost + DEA misses organised by shift |
| 🕳️ **Holes** | Packages that needed PS twice + costly failures |
| 📈 **Trend** | Track week-over-week (type numbers or upload CSVs) |
| 💾 **Export** | Download the data |
""")
    with st.expander("❓ What do the terms mean?"):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Effective** | The associate actually fixed the problem |
| **Ineffective** | They attempted to fix it but it didn't work |
| **SLA** | Did they fix it within the allowed time window? |
| **DEA Miss** | Package wasn't dispatched when it should have been |
| **Concession** | Money refunded to the customer because of the issue |
| **Process** | Where in the flow: Induct → Stow → Pick → Dispatch |
| **Cycle** | Which dispatch wave (CYCLE_1, HV_A, ADHOC, etc.) |
""")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

mode = st.radio("Mode:", ["📖 Guide", "Single Station"], horizontal=True, key="mode")

if mode == "📖 Guide":
    render_guide()

elif mode == "Single Station":
    c_pse, c_scc = st.columns(2)
    with c_pse: pse_file = st.file_uploader("🔧 PSE Dashboard CSV", type="csv", key="pse")
    with c_scc: scc_file = st.file_uploader("📋 SCC CSV (optional — for location data)", type="csv", key="scc")

    if pse_file:
        try: pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
        except Exception as e: st.error(f"❌ {e}"); st.stop()

        miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
        if miss: st.error(f"❌ Missing columns: {miss}"); st.stop()

        pse_df, removed = filter_uk_ids(pse_df)
        if removed > 0: st.info(f"Removed {removed} non-UK IDs — {len(pse_df)} kept.")

        scc_df = None
        if scc_file:
            try: scc_df = pd.read_csv(scc_file, encoding="utf-8-sig")
            except: scc_df = None

        df = clean_pse(pse_df)
        if scc_df is not None:
            df = merge_pse_scc(df, scc_df)
            matched = df["Cluster"].notna().sum() if "Cluster" in df.columns else 0
            st.success(f"✅ **{len(df)} events** — location data: {matched}/{len(df)}")
        else:
            for col in ["Cluster","Aisle","Sort Zone"]:
                if col not in df.columns: df[col] = None
            st.success(f"✅ **{len(df)} events** loaded")

        # FILTERS
        st.markdown("---")
        f1, f2 = st.columns(2)
        with f1:
            procs = sorted(df["Process"].dropna().unique().tolist())
            sel_procs = st.multiselect("Filter by Process:", procs, default=procs, key="f_proc")
        with f2:
            cats = sorted(df["Category"].dropna().unique().tolist())
            sel_cats = st.multiselect("Filter by Category:", cats, default=cats, key="f_cat")

        filtered = df.copy()
        if sel_procs: filtered = filtered[filtered["Process"].isin(sel_procs)]
        else: st.warning("Pick a process."); st.stop()
        if sel_cats: filtered = filtered[filtered["Category"].isin(sel_cats)]
        else: st.warning("Pick a category."); st.stop()

        total = len(filtered)
        if total == 0: st.warning("No events."); st.stop()

        # METRICS
        dr = get_date_range(filtered)
        eff = int(filtered["Is Effective"].sum())
        ineff = total - eff
        sla = int(filtered["SLA Met"].sum())
        cost = filtered["Cost (£)"].sum()

        st.markdown("---")
        if dr: st.caption(f"📅 **{dr}** | {total} events")

        score, color, label, reasons = compute_health_score(filtered, total)
        st.markdown(f"**Health: {color} {score}/10 — {label}**" + (f" ({', '.join(reasons)})" if reasons else ""))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Events", total)
        c2.metric("Effective", f"{eff} ({fmt_pct(eff,total)})")
        c3.metric("Ineffective", f"{ineff} ({fmt_pct(ineff,total)})")
        c4.metric("SLA Met", fmt_pct(sla,total))
        c5.metric("Concessions", fmt_cost(cost))

        # TABS
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "📊 Summary", "📍 Locations", "👤 Associates",
            "🔄 Cycles", "💰 Cost & DEA", "🕳️ Holes",
            "📈 Trend", "💾 Export"
        ])

        with t1: render_summary_tab(filtered, total, dr)
        with t2: render_locations_tab(filtered, total, dr)
        with t3: render_ps_tab(filtered, total, dr)
        with t4: render_cycles_tab(filtered, total, dr)
        with t5: render_cost_tab(filtered, total, dr)
        with t6: render_holes_tab(filtered, total, dr)
        with t7: render_trend_tab(filtered, total, dr)
        with t8: render_export_tab(filtered, total, dr)
    else:
        st.info("👆 Upload your PSE Dashboard CSV to get started.")
