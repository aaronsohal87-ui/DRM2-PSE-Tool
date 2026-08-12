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
    """Which shift does each solver mostly work?"""
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

def make_bar_horiz(data, title, color="steelblue"):
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

def make_eff_bar(df, group_col, title, top_n=15):
    """Stacked bar: green = effective, red = ineffective."""
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
    if "Resolution Time_DT" in df.columns: df["Resolution DT"] = df["Resolution Time_DT"]
    if "Resolution time taken(min)" in df.columns:
        df["Time to Fix (min)"] = pd.to_numeric(df["Resolution time taken(min)"].astype(str).str.replace(",",""), errors="coerce")
    else:
        df["Time to Fix (min)"] = float("nan")
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


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH SCORE
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ─── IDs for SCC ──────────────────────────────────────────────────────────
    with st.expander("📋 Tracking IDs — Copy into SCC for location data"):
        st.markdown("**Select all → Copy → Paste into SCC search box → Export SCC CSV → Upload here.**")
        uk_ids = sorted(df[df["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].astype(str).str.strip().unique())
        if len(uk_ids) > 0:
            st.caption(f"{len(uk_ids)} unique UK IDs (non-UK removed automatically)")
            st.code("\n".join(uk_ids), language=None)
        else:
            st.warning("No UK IDs in current data.")

    # ─── By Process ───────────────────────────────────────────────────────────
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

    # ─── By Category ──────────────────────────────────────────────────────────
    with st.expander("🏷️ By Category", expanded=True):
        st.pyplot(make_eff_bar(df, "Category", f"Categories ({dr})"))

    # ─── By Shift ─────────────────────────────────────────────────────────────
    with st.expander("⏰ By Shift"):
        sd = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        sd["Ineffective"] = sd["Total"] - sd["Effective"]
        sd["Eff %"] = (sd["Effective"]/sd["Total"]*100).round(1)
        sd["Window"] = [SHIFT_DEFINITIONS.get(s,"") for s in sd.index]
        st.dataframe(sd[["Total","Effective","Ineffective","Eff %","Window"]], use_container_width=True)
        st.pyplot(make_bar_shift(df[df["Shift"].isin(SHIFT_ORDER)]["Shift"].value_counts().reindex(SHIFT_ORDER, fill_value=0), f"By Shift ({dr})"))

    # ─── Hour of Day ──────────────────────────────────────────────────────────
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
# TAB: LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def render_locations_tab(df, total, dr):
    if total == 0: st.warning("No data."); return
    has_scc = "Cluster" in df.columns and df["Cluster"].notna().any()
    if not has_scc:
        st.warning("⚠️ No SCC uploaded — go to Summary tab → copy IDs → paste into SCC → export → upload here.")
        if "Route" in df.columns and df["Route"].notna().any():
            st.pyplot(make_eff_bar(df[df["Route"].notna()], "Route", f"By Route ({dr})", top_n=20))
        return

    with st.expander("📍 By Cluster", expanded=True):
        st.pyplot(make_eff_bar(df[df["Cluster"].notna()], "Cluster", f"By Cluster ({dr})"))

    with st.expander("🏷️ By Aisle"):
        if df["Aisle"].notna().any():
            st.pyplot(make_eff_bar(df[df["Aisle"].notna()], "Aisle", f"By Aisle ({dr})", top_n=20))

    with st.expander("🗂️ By Sort Zone"):
        if "Sort Zone" in df.columns and df["Sort Zone"].notna().any():
            st.pyplot(make_eff_bar(df[df["Sort Zone"].notna()], "Sort Zone", f"By Sort Zone ({dr})"))

    with st.expander("🔍 Drill into a Cluster"):
        clusters = sorted(df["Cluster"].dropna().unique().tolist())
        if clusters:
            sel = st.selectbox("Pick a cluster:", clusters, key="drill_cl")
            filt = df[df["Cluster"]==sel]
            e = int(filt["Is Effective"].sum())
            st.write(f"**{len(filt)} events** — {e} effective ({fmt_pct(e,len(filt))})")
            if filt["Aisle"].notna().any():
                st.pyplot(make_bar_horiz(filt["Aisle"].dropna().value_counts(), f"{sel} — Aisles"))
            st.pyplot(make_bar_horiz(filt["Category"].dropna().value_counts(), f"{sel} — Categories", color="purple"))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PROBLEM SOLVERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_ps_tab(df, total, dr):
    """
    Shows all problem solvers ranked from worst to best by Eff%.
    Also flags those with worst SLA%.
    Shift shown so managers know who to speak to and when.
    """
    if total == 0: st.warning("No data."); return

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

    # Only rank those with 3+ events
    ranked = ps[ps["Total"] >= 3].sort_values("Eff %", ascending=True).copy()

    with st.expander("👤 All Associates — Worst → Best (3+ events)", expanded=True):
        st.caption("Sorted by effectiveness % (lowest at top). Shift = their usual working shift. Talk to the people at the top.")
        if len(ranked) > 0:
            avg_eff = ranked["Eff %"].mean()
            avg_sla = ranked["SLA %"].mean()
            st.markdown(f"**Team average: {avg_eff:.0f}% effective | {avg_sla:.0f}% SLA**")
            display = ranked[["Shift","Total","Effective","Ineffective","Eff %","SLA %"]].reset_index()
            display = display.rename(columns={"PS Display":"Associate"})
            display.index = range(1, len(display)+1)
            display.index.name = "Rank"
            st.dataframe(display, use_container_width=True, height=min(700, 35*len(display)+40))

    with st.expander("🔴 Flagged — Need Coaching (Eff % below average)", expanded=True):
        if len(ranked) >= 3:
            avg_eff = ranked["Eff %"].mean()
            flagged = ranked[ranked["Eff %"] < avg_eff - 10]
            if len(flagged) > 0:
                st.error(f"🚨 {len(flagged)} associate(s) more than 10pp below average ({avg_eff:.0f}%):")
                for name, row in flagged.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']} shift]: **{row['Eff %']}%** effective ({int(row['Ineffective'])} ineffective out of {int(row['Total'])})")
            else:
                st.success(f"✅ No one significantly below average ({avg_eff:.0f}%).")

    with st.expander("🔴 Flagged — Worst SLA % (slowest to resolve)"):
        st.caption("SLA = did they resolve the problem within the allowed time? These associates are the slowest.")
        if len(ranked) >= 3:
            avg_sla = ranked["SLA %"].mean()
            flagged_sla = ranked[ranked["SLA %"] < avg_sla - 10]
            if len(flagged_sla) > 0:
                st.error(f"🚨 {len(flagged_sla)} associate(s) more than 10pp below SLA average ({avg_sla:.0f}%):")
                for name, row in flagged_sla.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']} shift]: **{row['SLA %']}%** SLA ({int(row['Total'])} events)")
            else:
                st.success(f"✅ No one significantly below SLA average ({avg_sla:.0f}%).")

    with st.expander("📊 What does each associate handle?"):
        st.caption("Shows which processes each person works on and their effectiveness for each.")
        ps_proc = df.groupby(["PS Display","Process"]).agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).reset_index()
        ps_proc["Eff %"] = (ps_proc["Effective"]/ps_proc["Total"]*100).round(1)
        pivot = ps_proc.pivot_table(index="PS Display", columns="Process", values="Total", fill_value=0)
        valid = pivot[pivot.sum(axis=1) >= 3].sort_values(by=pivot.columns.tolist(), ascending=False)
        if len(valid) > 0:
            st.markdown("**Event count by associate × process:**")
            st.dataframe(valid.astype(int), use_container_width=True)

    with st.expander("🎯 Where are they failing? (Category × Associate)"):
        st.caption("Which categories each associate is worst at. Only shows combos with 3+ events.")
        ps_cat = df.groupby(["PS Display","Category"]).agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum")
        ).reset_index()
        ps_cat["Eff %"] = (ps_cat["Effective"]/ps_cat["Total"]*100).round(1)
        worst_combos = ps_cat[ps_cat["Total"]>=3].sort_values("Eff %", ascending=True).head(20)
        if len(worst_combos) > 0:
            worst_combos["Ineffective"] = worst_combos["Total"] - worst_combos["Effective"]
            out = worst_combos[["PS Display","Category","Total","Ineffective","Eff %"]].rename(columns={"PS Display":"Associate"})
            out.index = range(1, len(out)+1)
            st.dataframe(out, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: TIME & CYCLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_time_tab(df, total, dr):
    if total == 0: st.warning("No data."); return

    with st.expander("🔄 By Cycle", expanded=True):
        if "Actual Cycle" in df.columns:
            cyc = df.groupby("Actual Cycle").agg(Total=("Scannable ID","count"), Effective=("Is Effective","sum")).sort_values("Total", ascending=False)
            cyc["Eff %"] = (cyc["Effective"]/cyc["Total"]*100).round(1)
            st.dataframe(cyc[["Total","Effective","Eff %"]], use_container_width=True)
            st.pyplot(make_bar_horiz(df["Actual Cycle"].dropna().value_counts(), f"By Cycle ({dr})", color="teal"))

    with st.expander("⏰ Shift Table"):
        st.caption("NS: 23:45–09:45 | AM: 09:45–14:00 | PM: 14:00–23:45")
        se = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
            Total=("Scannable ID","count"), Effective=("Is Effective","sum"), SLA=("SLA Met","sum")
        ).reindex(SHIFT_ORDER, fill_value=0)
        se["Eff %"] = (se["Effective"]/se["Total"]*100).round(1)
        se["SLA %"] = (se["SLA"]/se["Total"]*100).round(1)
        se["Window"] = [SHIFT_DEFINITIONS.get(s,"") for s in se.index]
        st.dataframe(se[["Total","Effective","Eff %","SLA %","Window"]], use_container_width=True)

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
    c2.metric("Events with Cost", with_cost)
    c3.metric("DEA Misses", dea)

    with st.expander("💰 Cost by Category", expanded=True):
        cdf = df[df["Cost (£)"]>0]
        if len(cdf) > 0:
            cc = cdf.groupby("Category").agg(Events=("Scannable ID","count"), Cost=("Cost (£)","sum")).sort_values("Cost", ascending=False).reset_index()
            cc["Avg"] = (cc["Cost"]/cc["Events"]).apply(fmt_cost); cc["Cost"] = cc["Cost"].apply(fmt_cost)
            cc.index = range(1,len(cc)+1); st.dataframe(cc, use_container_width=True)

    with st.expander("💰 Cost of Ineffective PS"):
        st.caption("How much money is lost when PS fails? This is concession cost on events where the problem solver was NOT effective.")
        ineff_cost = df[~df["Is Effective"]]["Cost (£)"].sum()
        eff_cost = df[df["Is Effective"]]["Cost (£)"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Cost (Effective)", fmt_cost(eff_cost))
        c2.metric("Cost (Ineffective)", fmt_cost(ineff_cost))
        if ineff_cost > 0:
            st.error(f"💡 **{fmt_cost(ineff_cost)}** lost on events where PS failed. This is preventable.")

    with st.expander("🎯 DEA Misses"):
        dea_events = df[df["DEA Miss"]>0]
        if len(dea_events) > 0:
            st.error(f"🚨 {len(dea_events)} event(s) with DEA misses")
            if "dea_bucket" in df.columns:
                db = dea_events["dea_bucket"].dropna().value_counts()
                if len(db)>0: st.pyplot(make_bar_horiz(db, "DEA Buckets", color="darkred"))
            cols = [c for c in ["Scannable ID","Process","Category","PS Display","Shift","dea_bucket"] if c in dea_events.columns]
            st.dataframe(dea_events[cols].reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No DEA misses.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HOLES
# ═══════════════════════════════════════════════════════════════════════════════

def render_holes_tab(df, total, dr):
    if total == 0: st.warning("No data."); return
    st.markdown("### 🕳️ Finding Holes in Problem Solve")
    st.caption("Systemic issues — things that keep going wrong or aren't being fixed properly.")

    with st.expander("🔁 Same Package, Multiple PS Events", expanded=True):
        st.caption("If a package was problem-solved more than once, nobody actually fixed it the first time.")
        id_counts = df.groupby("Scannable ID").size()
        repeats = id_counts[id_counts > 1].sort_values(ascending=False)
        if len(repeats) > 0:
            st.error(f"🚨 **{len(repeats)} packages** needed PS more than once!")
            rdf = df[df["Scannable ID"].isin(repeats.index)].sort_values(["Scannable ID","Exception Open DT"])
            cols = [c for c in ["Scannable ID","Process","Category","Effective","PS Display","Shift","Status"] if c in rdf.columns]
            st.dataframe(rdf[cols].reset_index(drop=True), use_container_width=True, height=300)
        else:
            st.success("✅ No repeat PS events.")

    with st.expander("⏳ Never Resolved"):
        st.caption("Events with no resolution recorded — may have been abandoned or forgotten.")
        unresolved = df[df["Time to Fix (min)"].isna()]
        if len(unresolved) > 0:
            st.warning(f"⚠️ **{len(unresolved)} events** ({fmt_pct(len(unresolved), total)}) have no resolution.")
            ur_ps = unresolved["PS Display"].value_counts().head(10)
            st.markdown("**Who leaves events unresolved:**")
            tbl = ur_ps.reset_index(); tbl.columns = ["Associate","Unresolved"]; tbl.index = range(1,len(tbl)+1)
            st.dataframe(tbl, use_container_width=True)
        else:
            st.success("✅ All events resolved.")

    with st.expander("❌ SLA Misses — Who & What"):
        st.caption("SLA = did they fix the problem within the allowed time window? These events missed it.")
        sla_miss = df[~df["SLA Met"]]
        if len(sla_miss) > 0:
            st.warning(f"⚠️ **{len(sla_miss)} SLA misses** ({fmt_pct(len(sla_miss), total)})")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**By Associate:**")
                sp = sla_miss["PS Display"].value_counts().head(10)
                tbl = sp.reset_index(); tbl.columns = ["Associate","Misses"]; tbl.index = range(1,len(tbl)+1)
                st.dataframe(tbl, use_container_width=True)
            with c2:
                st.markdown("**By Category:**")
                sc = sla_miss["Category"].value_counts().head(10)
                tbl2 = sc.reset_index(); tbl2.columns = ["Category","Misses"]; tbl2.index = range(1,len(tbl2)+1)
                st.dataframe(tbl2, use_container_width=True)
        else:
            st.success("✅ No SLA misses.")

    with st.expander("📉 Ineffective + Concession Cost"):
        st.caption("The worst outcomes — PS failed AND it cost money (customer concession).")
        bad = df[(~df["Is Effective"]) & (df["Cost (£)"]>0)].sort_values("Cost (£)", ascending=False)
        if len(bad) > 0:
            st.error(f"🚨 **{len(bad)} events** failed PS + cost money (total: {fmt_cost(bad['Cost (£)'].sum())})")
            cols = [c for c in ["Scannable ID","Process","Category","PS Display","Shift","Cost (£)"] if c in bad.columns]
            st.dataframe(bad[cols].head(20).reset_index(drop=True), use_container_width=True)
        else:
            st.success("✅ No costly failures.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: ANALYSIS & TREND
# ═══════════════════════════════════════════════════════════════════════════════

def render_analysis_tab(df, total, dr):
    if total == 0: st.warning("No data."); return
    view = st.selectbox("View:", ["🔬 Analysis", "📈 Trend"], key="at_view")

    if view == "🔬 Analysis":
        st.markdown("### 🔬 Key Findings")
        st.caption("Data-driven — use your own judgement.")
        actions = []

        with st.expander("📍 Where are problems?", expanded=True):
            has_scc = "Cluster" in df.columns and df["Cluster"].notna().any()
            if has_scc:
                cl = df["Cluster"].dropna().value_counts()
                if len(cl) >= 2:
                    top5 = cl.head(5)
                    pct = round(top5.sum()/cl.sum()*100,1)
                    st.markdown(f"**Top 5 clusters = {pct}% of all events**")
                    for name, count in top5.items():
                        cdf = df[df["Cluster"]==name]
                        eff = cdf["Is Effective"].mean()*100
                        cat = cdf["Category"].value_counts().index[0] if len(cdf["Category"].dropna())>0 else "?"
                        st.markdown(f"- **{name}**: {int(count)} events, {eff:.0f}% eff — mainly: {cat}")
                    actions.append(f"Walk {', '.join(top5.index[:3])}")
            else:
                st.info("Upload SCC for location analysis.")

        with st.expander("📊 Worst categories"):
            ce = df.groupby("Category").agg(Total=("Scannable ID","count"), Effective=("Is Effective","sum")).sort_values("Total", ascending=False)
            ce["Eff %"] = (ce["Effective"]/ce["Total"]*100).round(1)
            worst = ce[ce["Total"]>=5].sort_values("Eff %").head(3)
            if len(worst)>0:
                for name, row in worst.iterrows():
                    st.markdown(f"- **{name}**: {row['Eff %']}% eff ({int(row['Total'])} events)")
                actions.append(f"Investigate '{worst.index[0]}'")

        with st.expander("⏰ Worst shift"):
            sc = df[df["Shift"].isin(SHIFT_ORDER)].groupby("Shift").agg(
                Total=("Scannable ID","count"), Effective=("Is Effective","sum")
            ).reindex(SHIFT_ORDER, fill_value=0)
            sc["Eff %"] = (sc["Effective"]/sc["Total"]*100).round(1)
            for s in SHIFT_ORDER:
                m = "🔴" if sc.loc[s,"Eff %"]<60 else "🟡" if sc.loc[s,"Eff %"]<75 else "🟢"
                st.markdown(f"  {m} **{s}**: {sc.loc[s,'Eff %']}% ({int(sc.loc[s,'Total'])} events)")

        if actions:
            st.markdown("---")
            st.markdown("#### 📋 Actions")
            for i, a in enumerate(actions, 1): st.markdown(f"**{i}.** {a}")

    else:
        st.markdown("### 📈 Week-over-Week Trend")
        st.caption("Upload one PSE CSV per week OR type values.")
        tm = st.selectbox("Input:", ["📝 Type values", "📂 Upload CSVs"], key="tm")
        if tm == "📝 Type values":
            nw = st.slider("Weeks:", 2, 12, 4, key="tw_n")
            weeks = []
            for i in range(nw):
                with st.expander(f"Week {i+1}", expanded=(i<2)):
                    wl = st.text_input("Label:", value=f"W{i+1}", key=f"tw_l{i}")
                    wt = st.number_input("Total events:", min_value=0, value=0, step=1, key=f"tw_t{i}")
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
                        we = int((wdf.get("Effective (Y/N)","").astype(str).str.strip().str.upper()=="Y").sum()) if "Effective (Y/N)" in wdf.columns else 0
                        weeks.append({"Week":wl,"Total":len(wdf),"Effective":we})
                        st.caption(f"→ {wl}: {len(wdf)} events, {we} effective")
                    except Exception as e: st.error(f"Error: {e}")
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
    elif len(weeks)==1: st.info("Need 2+ weeks.")


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
**Step 1:** Export PSE Dashboard → Raw Data → CSV

**Step 2:** Upload it here

**Step 3:** (Optional) Go to Summary tab → copy the tracking IDs → paste into SCC → export SCC CSV → upload here too

**That's it.** Use the tabs to explore:
- **Summary** — overview of everything
- **Locations** — where problems happen (needs SCC)
- **Problem Solvers** — who's effective, who needs help
- **Time & Cycles** — when problems happen
- **Cost & DEA** — money impact
- **Holes** — systemic issues nobody's fixing
- **Analysis & Trend** — week-over-week tracking
""")
    with st.expander("❓ What do the terms mean?"):
        st.markdown("""
| Term | What it means |
|------|--------------|
| **Effective** | The problem solver actually fixed the issue correctly |
| **Ineffective** | They attempted to fix it but it didn't work |
| **SLA** | Did they fix it within the time limit? (Service Level Agreement) |
| **DEA Miss** | A dispatch error — package wasn't dispatched when it should have been |
| **Concession** | Money Amazon had to refund the customer because of the issue |
| **Process** | Where in the station flow it happened: Induct → Stow → Pick → Dispatch |
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
    with c_scc: scc_file = st.file_uploader("📋 SCC CSV (optional — for location drill-down)", type="csv", key="scc")

    if pse_file:
        try: pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
        except Exception as e: st.error(f"❌ {e}"); st.stop()

        miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
        if miss: st.error(f"❌ Missing: {miss}"); st.stop()

        # Remove non-UK IDs
        pse_df, removed = filter_uk_ids(pse_df)
        if removed > 0: st.info(f"Removed {removed} non-UK IDs — {len(pse_df)} kept.")

        # SCC
        scc_df = None
        if scc_file:
            try: scc_df = pd.read_csv(scc_file, encoding="utf-8-sig")
            except: scc_df = None

        # Clean & Merge
        df = clean_pse(pse_df)
        if scc_df is not None:
            df = merge_pse_scc(df, scc_df)
            matched = df["Cluster"].notna().sum() if "Cluster" in df.columns else 0
            st.success(f"✅ **{len(df)} events** — location data: {matched}/{len(df)}")
        else:
            for col in ["Cluster","Aisle","Sort Zone"]:
                if col not in df.columns: df[col] = None
            st.success(f"✅ **{len(df)} events** loaded")

        # ─── FILTERS (simplified: Process + Category only) ───────────────────
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
        else: st.warning("Pick at least one process."); st.stop()
        if sel_cats: filtered = filtered[filtered["Category"].isin(sel_cats)]
        else: st.warning("Pick at least one category."); st.stop()

        total = len(filtered)
        if total == 0: st.warning("No events match."); st.stop()

        # ─── METRICS ─────────────────────────────────────────────────────────
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
        c5.metric("Concession Cost", fmt_cost(cost))

        # ─── TABS ────────────────────────────────────────────────────────────
        st.markdown("---")
        t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
            "📊 Summary", "📍 Locations", "👤 Problem Solvers",
            "⏰ Time & Cycles", "💰 Cost & DEA", "🕳️ Holes",
            "🔬 Analysis & Trend", "💾 Export"
        ])

        with t1: render_summary_tab(filtered, total, dr)
        with t2: render_locations_tab(filtered, total, dr)
        with t3: render_ps_tab(filtered, total, dr)
        with t4: render_time_tab(filtered, total, dr)
        with t5: render_cost_tab(filtered, total, dr)
        with t6: render_holes_tab(filtered, total, dr)
        with t7: render_analysis_tab(filtered, total, dr)
        with t8: render_export_tab(filtered, total, dr)
    else:
        st.info("👆 Upload your PSE Dashboard CSV to get started.")
