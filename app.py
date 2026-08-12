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
LABEL_MAX = 30; CHART = (7, 2.5)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def hour_to_shift(h):
    if pd.isna(h): return "Unknown"
    try: return SHIFT_HOUR_MAP.get(int(h), "Unknown")
    except: return "Unknown"

def assign_shift(row):
    for col in ["Exception Open DT", "PSS Event DT"]:
        v = row.get(col)
        if pd.notna(v):
            try: return hour_to_shift(v.hour)
            except: pass
    return "Unknown"

def get_solver_shift(df):
    return df.groupby("PS Display")["Shift"].agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else "?").to_dict()

def fmt_pct(n, d):
    return f"{round(n/d*100,1)}%" if d>0 else "0%"

def fmt_cost(v):
    try:
        if pd.isna(v): return "£0.00"
        return f"£{float(v):,.2f}"
    except: return "£0.00"

def trunc(labels, mx=LABEL_MAX):
    return [str(l)[:mx]+"..." if len(str(l))>mx else str(l) for l in labels]

def get_date_range(df):
    if "Date" in df.columns:
        d = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce").dropna()
        if len(d)>0:
            s,e = d.min().strftime("%d %b %Y"), d.max().strftime("%d %b %Y")
            return s if s==e else f"{s} – {e}"
    return ""

def make_bar_horiz(data, title, color="steelblue", n=10):
    data = data.head(n)
    if len(data)==0: return plt.subplots(figsize=(7,2))[0]
    h = max(2, len(data)*0.3); fig, ax = plt.subplots(figsize=(7,h))
    ax.barh(trunc(data.index), data.values, color=color); ax.invert_yaxis()
    mx = data.values.max() if len(data)>0 else 1; ax.set_xlim(right=mx*1.18)
    for i,v in enumerate(data.values): ax.text(v+mx*0.02, i, str(int(v)), va="center", fontsize=7)
    ax.set_xlabel("Count",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_bar_shift(data, title):
    data = data.reindex(SHIFT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=CHART)
    bars = ax.bar(SHIFT_ORDER, [data[s] for s in SHIFT_ORDER], color=[SHIFT_COLORS[s] for s in SHIFT_ORDER])
    mx = max(data[s] for s in SHIFT_ORDER) if any(data[s]>0 for s in SHIFT_ORDER) else 1
    ax.set_ylim(top=mx*1.25)
    for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=7)
    ax.set_xlabel("Shift",fontsize=8); ax.set_ylabel("Count",fontsize=8); ax.set_title(title,fontsize=9)
    ax.tick_params(labelsize=7); plt.tight_layout()
    return fig

def make_eff_bar(df, col, title, n=10):
    g = df.groupby(col).agg(Total=("Scannable ID","count"), Effective=("Is Effective","sum")).sort_values("Total",ascending=False).head(n)
    g["Ineffective"] = g["Total"]-g["Effective"]; g["Eff %"] = (g["Effective"]/g["Total"]*100).round(1)
    if len(g)==0: fig,ax=plt.subplots(figsize=(7,2)); ax.text(0.5,0.5,"No data",ha="center"); return fig
    h = max(2, len(g)*0.35); fig,ax = plt.subplots(figsize=(7,h))
    ax.barh(trunc(g.index), g["Effective"].values, color="#2ecc71", label="Effective")
    ax.barh(trunc(g.index), g["Ineffective"].values, left=g["Effective"].values, color="#e74c3c", label="Ineffective")
    ax.invert_yaxis(); mx=g["Total"].max(); ax.set_xlim(right=mx*1.25)
    for i,(t,r) in enumerate(zip(g["Total"].values, g["Eff %"].values)): ax.text(t+mx*0.02, i, f"{int(t)} ({r}%)", va="center", fontsize=7)
    ax.set_xlabel("Events",fontsize=8); ax.set_title(title,fontsize=9); ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
def clean_pse(df):
    df = df.copy(); df.columns = df.columns.str.strip()
    for c in ["Exception Open Time","Resolution Time","PSS Event Time","Shipment Status Datetime"]:
        if c in df.columns: df[c+"_DT"] = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
    if "Exception Open Time_DT" in df.columns: df["Exception Open DT"] = df["Exception Open Time_DT"]
    if "PSS Event Time_DT" in df.columns: df["PSS Event DT"] = df["PSS Event Time_DT"]
    if "gross_concession" in df.columns:
        df["Cost (£)"] = pd.to_numeric(df["gross_concession"].astype(str).str.replace("[£$,]","",regex=True), errors="coerce").fillna(0)
    else: df["Cost (£)"] = 0.0
    df["Shift"] = df.apply(assign_shift, axis=1)
    df["Effective"] = df["Effective (Y/N)"].astype(str).str.strip().str.upper()
    df["Is Effective"] = df["Effective"]=="Y"
    df["SLA Met"] = df["SLA (Y/N)"].astype(str).str.strip().str.upper()=="Y" if "SLA (Y/N)" in df.columns else False
    df["PS Display"] = df["Problem_Solver"].astype(str).str.replace("@amazon.com","",regex=False).str.strip() if "Problem_Solver" in df.columns else "Unknown"
    if "Process" in df.columns: df["Process"] = df["Process"].astype(str).str.strip().str.upper()
    if "Category" in df.columns: df["Category"] = df["Category"].astype(str).str.strip()
    df["DEA Miss"] = pd.to_numeric(df.get("dea_miss", 0), errors="coerce").fillna(0).astype(int)
    return df

def clean_scc(df):
    df = df.copy(); df.columns = df.columns.str.strip()
    if "Tracking ID" in df.columns: df["Tracking ID"] = df["Tracking ID"].astype(str).str.strip()
    return df[[c for c in ["Tracking ID","Sort Zone","Aisle","Cluster"] if c in df.columns]]

def merge_pse_scc(pse, scc):
    pse = pse.copy(); scc = clean_scc(scc.copy())
    pse["_k"] = pse["Scannable ID"].astype(str).str.strip()
    scc["_k"] = scc["Tracking ID"].astype(str).str.strip()
    m = pse.merge(scc, on="_k", how="left", suffixes=("","_scc"))
    return m.drop(columns=["_k"], errors="ignore")

def filter_uk_ids(df):
    mask = df["Scannable ID"].astype(str).str.strip().str.startswith("UK")
    return df[mask].copy(), (~mask).sum()

def compute_health(df, total):
    if total==0: return 5,"🟡","No data",[]
    score=10; reasons=[]
    e = df["Is Effective"].sum()/total
    if e<0.5: score-=3; reasons.append(f"Eff {e*100:.0f}%")
    elif e<0.65: score-=2; reasons.append(f"Eff {e*100:.0f}%")
    elif e<0.75: score-=1; reasons.append(f"Eff {e*100:.0f}%")
    s = df["SLA Met"].sum()/total
    if s<0.5: score-=2; reasons.append(f"SLA {s*100:.0f}%")
    elif s<0.7: score-=1; reasons.append(f"SLA {s*100:.0f}%")
    d = df["DEA Miss"].sum()
    if d>=5: score-=2; reasons.append(f"{int(d)} DEA misses")
    elif d>=2: score-=1; reasons.append(f"{int(d)} DEA miss(es)")
    score = max(1,min(10,score))
    if score>=8: return score,"🟢","Good",reasons
    elif score>=5: return score,"🟡","Needs attention",reasons
    return score,"🔴","Action required",reasons

# ═══════════════════════════════════════════════════════════════════════════════
# RENDER FUNCTIONS (used by both single + multi-station)
# ═══════════════════════════════════════════════════════════════════════════════

def render_summary(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    with st.expander("📋 Tracking IDs — Copy into SCC", expanded=False):
        st.markdown("**Copy IDs below → open SCC → upload with all options selected → export CSV → upload that here as SCC file.**")
        uk = sorted(df[df["Scannable ID"].astype(str).str.startswith("UK")]["Scannable ID"].astype(str).str.strip().unique())
        if len(uk)>0:
            st.caption(f"{len(uk)} unique UK IDs (non-UK removed automatically)")
            st.code("\n".join(uk), language=None)
        else: st.warning("No UK IDs.")

    with st.expander("📦 By Process", expanded=True):
        p = df.groupby("Process").agg(Total=("Scannable ID","count"),Effective=("Is Effective","sum"),SLA=("SLA Met","sum")).reindex(PROCESS_ORDER,fill_value=0)
        p["Ineffective"]=p["Total"]-p["Effective"]; p["Eff %"]=(p["Effective"]/p["Total"]*100).round(1); p["SLA %"]=(p["SLA"]/p["Total"]*100).round(1)
        c1,c2=st.columns([1,1])
        with c1: st.dataframe(p[["Total","Effective","Ineffective","Eff %","SLA %"]], use_container_width=True)
        with c2:
            pt=p["Total"]; pt=pt[pt>0]
            if len(pt)>0:
                fig,ax=plt.subplots(figsize=(3,2.5))
                ax.pie(pt.values,labels=pt.index,colors=[PROCESS_COLORS.get(x,"gray") for x in pt.index],autopct="%1.0f%%",startangle=90,textprops={"fontsize":7})
                ax.set_title(f"By Process ({dr})",fontsize=8); plt.tight_layout(); st.pyplot(fig)

    with st.expander("🏷️ By Category", expanded=True):
        st.pyplot(make_eff_bar(df, "Category", f"Categories ({dr})"))

    with st.expander("🕐 Hour of Day"):
        if "Exception Open DT" in df.columns:
            hours = df["Exception Open DT"].dropna().dt.hour
            if len(hours)>0:
                hc = hours.value_counts().sort_index().reindex(range(24), fill_value=0)
                fig,ax = plt.subplots(figsize=(8,2.5))
                ax.bar(range(24), hc.values, color=[SHIFT_COLORS.get(SHIFT_HOUR_MAP.get(h,""),"gray") for h in range(24)])
                for h in range(24):
                    if hc.values[h]>0: ax.text(h, hc.values[h]+0.1, str(int(hc.values[h])), ha="center", fontsize=6)
                ax.set_xlabel("Hour",fontsize=8); ax.set_ylabel("Events",fontsize=8)
                ax.set_xticks(range(24)); ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)
                st.caption("🟦 NS (23:45–09:45) | 🟧 AM (09:45–14:00) | 🟩 PM (14:00–23:45)")


def render_locations(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    has_scc = "Cluster" in df.columns and df["Cluster"].notna().any()
    if not has_scc:
        st.warning("⚠️ No SCC data. Go to Summary → copy IDs → upload into SCC with all options selected → export CSV → upload here.")
        return

    with st.expander("📍 Cluster Overview (top 10)", expanded=True):
        st.pyplot(make_eff_bar(df[df["Cluster"].notna()], "Cluster", f"Top 10 Clusters ({dr})"))

    with st.expander("🗂️ Worst Sort Zones"):
        if "Sort Zone" in df.columns and df["Sort Zone"].notna().any():
            sz = df[df["Sort Zone"].notna()].groupby("Sort Zone").agg(
                Events=("Scannable ID","count"), Effective=("Is Effective","sum")
            ).sort_values("Events", ascending=False)
            sz["Ineffective"] = sz["Events"]-sz["Effective"]
            sz["Eff %"] = (sz["Effective"]/sz["Events"]*100).round(1)
            # Add top category per sort zone
            top_cats = df[df["Sort Zone"].notna()].groupby("Sort Zone")["Category"].agg(lambda x: x.value_counts().index[0] if len(x)>0 else "")
            sz["Main Issue"] = sz.index.map(top_cats)
            st.dataframe(sz[["Events","Ineffective","Eff %","Main Issue"]], use_container_width=True)
        else: st.info("No sort zone data.")

    st.markdown("### 🔍 Cluster Deep Dive")
    st.caption("Select a cluster to see aisles, categories, and shifts inside it.")
    clusters = df["Cluster"].dropna().value_counts()
    if len(clusters)>0:
        sel = st.selectbox("Cluster:", clusters.index.tolist(), format_func=lambda x: f"{x} ({int(clusters[x])} events)", key=f"{kp}drill")
        filt = df[df["Cluster"]==sel]
        e = int(filt["Is Effective"].sum()); ie = len(filt)-e
        c1,c2,c3 = st.columns(3)
        c1.metric("Events", len(filt)); c2.metric("Effective", f"{e} ({fmt_pct(e,len(filt))})"); c3.metric("Ineffective", f"{ie}")

        st.markdown("**Aisles:**")
        if "Aisle" in filt.columns and filt["Aisle"].notna().any():
            a = filt.groupby("Aisle").agg(Events=("Scannable ID","count"),Effective=("Is Effective","sum")).sort_values("Events",ascending=False)
            a["Ineffective"]=a["Events"]-a["Effective"]; a["Eff %"]=(a["Effective"]/a["Events"]*100).round(1)
            st.dataframe(a[["Events","Effective","Ineffective","Eff %"]], use_container_width=True)

        st.markdown("**What's going wrong:**")
        cat = filt.groupby("Category").agg(Events=("Scannable ID","count"),Effective=("Is Effective","sum")).sort_values("Events",ascending=False)
        cat["Eff %"] = (cat["Effective"]/cat["Events"]*100).round(1)
        st.dataframe(cat[["Events","Effective","Eff %"]], use_container_width=True)

        st.markdown("**Which shift:**")
        sh = filt[filt["Shift"].isin(SHIFT_ORDER)].groupby("Shift").size().reindex(SHIFT_ORDER, fill_value=0)
        st.pyplot(make_bar_shift(sh, f"{sel} — By Shift"))


def render_associates(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    st.markdown("""
> ⚠️ **Data source & accuracy**
>
> All data below comes directly from the **PSE Dashboard CSV export** (raw data).
> - **Associate** = `Problem_Solver` column (the login who handled the PS event)
> - **Eff %** = count of rows where `Effective (Y/N) = Y` ÷ total rows for that associate × 100
> - **SLA %** = count of rows where `SLA (Y/N) = Y` ÷ total rows for that associate × 100
> - **Shift** = most common shift based on `Exception Open Time` hour
>
> **⚠️ Before taking any action:** Verify with the associate directly. Shared logins,
> system misattributions, or data errors can occur. This tool identifies patterns for
> investigation — it is not proof of performance issues.
""")
    shifts = get_solver_shift(df)
    ps = df.groupby("PS Display").agg(Total=("Scannable ID","count"),Effective=("Is Effective","sum"),SLA=("SLA Met","sum"),Cost=("Cost (£)","sum"))
    ps["Ineffective"]=ps["Total"]-ps["Effective"]; ps["Eff %"]=(ps["Effective"]/ps["Total"]*100).round(1)
    ps["SLA %"]=(ps["SLA"]/ps["Total"]*100).round(1); ps["Shift"]=ps.index.map(lambda x: shifts.get(x,"?"))
    ranked = ps[ps["Total"]>=3].sort_values("Eff %", ascending=True)

    with st.expander("👤 All Associates — Ranked Worst → Best (3+ events)", expanded=True):
        st.caption("**0% = never resolved a single package.** These are at the top.")
        if len(ranked)>0:
            avg_e = ranked["Eff %"].mean(); avg_s = ranked["SLA %"].mean()
            st.markdown(f"**Team average: {avg_e:.0f}% Eff | {avg_s:.0f}% SLA** (across {len(ranked)} associates with 3+ events)")
            zero = ranked[ranked["Eff %"]==0]
            if len(zero)>0:
                st.error(f"🚨 {len(zero)} associate(s) with **0% effectiveness**:")
                for name,row in zero.iterrows():
                    st.markdown(f"- **{name}** [{row['Shift']}] — {int(row['Total'])} events, none resolved")
            disp = ranked[["Shift","Total","Effective","Ineffective","Eff %","SLA %"]].reset_index().rename(columns={"PS Display":"Associate"})
            disp.index = range(1,len(disp)+1); disp.index.name = "Rank"
            st.dataframe(disp, use_container_width=True, height=min(700, 35*len(disp)+40))

    with st.expander("🔴 Flagged — Below Average"):
        if len(ranked)>=3:
            avg_e = ranked["Eff %"].mean(); avg_s = ranked["SLA %"].mean()
            st.markdown("**Eff % below average:**")
            fe = ranked[ranked["Eff %"]<avg_e-10]
            if len(fe)>0:
                for n,r in fe.iterrows(): st.markdown(f"- **{n}** [{r['Shift']}]: **{r['Eff %']}%** ({int(r['Total'])} events)")
            else: st.success(f"✅ Nobody >10pp below Eff avg ({avg_e:.0f}%).")
            st.markdown("**SLA % below average:**")
            fs = ranked[ranked["SLA %"]<avg_s-10]
            if len(fs)>0:
                for n,r in fs.iterrows(): st.markdown(f"- **{n}** [{r['Shift']}]: **{r['SLA %']}%** SLA ({int(r['Total'])} events)")
            else: st.success(f"✅ Nobody >10pp below SLA avg ({avg_s:.0f}%).")

    with st.expander("🔁 Repeat Offenders — Associates whose first fix didn't stick"):
        st.caption("These associates first handled a package that later needed PS again. Their initial resolution didn't hold.")
        id_counts = df.groupby("Scannable ID").size()
        repeats = id_counts[id_counts>1]
        if len(repeats)>0:
            rdf = df[df["Scannable ID"].isin(repeats.index)].sort_values(["Scannable ID","Exception Open DT"])
            first = rdf.groupby("Scannable ID").first()["PS Display"]
            first_counts = first.value_counts()
            tbl = first_counts.reset_index(); tbl.columns = ["Associate","Packages That Came Back"]
            tbl["Shift"] = tbl["Associate"].map(shifts)
            tbl.index = range(1,len(tbl)+1)
            st.dataframe(tbl.head(15), use_container_width=True)
            st.caption(f"Data: {len(repeats)} packages had 2+ PS events. This shows who handled the FIRST attempt.")
        else: st.success("✅ No repeat packages.")

    with st.expander("🎯 Category breakdown per associate"):
        st.caption("Which categories each associate fails at. Only combos with 3+ events shown.")
        pc = df.groupby(["PS Display","Category"]).agg(Total=("Scannable ID","count"),Effective=("Is Effective","sum")).reset_index()
        pc["Eff %"] = (pc["Effective"]/pc["Total"]*100).round(1)
        worst = pc[pc["Total"]>=3].sort_values("Eff %", ascending=True).head(20)
        if len(worst)>0:
            worst["Ineffective"] = worst["Total"]-worst["Effective"]
            out = worst[["PS Display","Category","Total","Ineffective","Eff %"]].rename(columns={"PS Display":"Associate"})
            out.index = range(1,len(out)+1); st.dataframe(out, use_container_width=True)


def render_cycles(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    with st.expander("🔄 By Actual Cycle", expanded=True):
        if "Actual Cycle" in df.columns:
            c = df.groupby("Actual Cycle").agg(Total=("Scannable ID","count"),Effective=("Is Effective","sum"),SLA=("SLA Met","sum")).sort_values("Total",ascending=False)
            c["Eff %"]=(c["Effective"]/c["Total"]*100).round(1); c["SLA %"]=(c["SLA"]/c["Total"]*100).round(1)
            st.dataframe(c[["Total","Effective","Eff %","SLA %"]], use_container_width=True)
    with st.expander("📅 By Planned Cycle"):
        if "Planned Cycle" in df.columns:
            pc = df.groupby("Planned Cycle").agg(Total=("Scannable ID","count"),Effective=("Is Effective","sum")).sort_values("Total",ascending=False)
            pc["Eff %"]=(pc["Effective"]/pc["Total"]*100).round(1)
            st.dataframe(pc[["Total","Effective","Eff %"]], use_container_width=True)


def render_cost(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    tc = df["Cost (£)"].sum(); wc=(df["Cost (£)"]>0).sum(); dea=int(df["DEA Miss"].sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Customer Refunds", fmt_cost(tc)); c2.metric("Events with Refund", wc); c3.metric("DEA Misses", dea)

    with st.expander("ℹ️ Where does this cost come from?", expanded=False):
        st.markdown("""
**This is NOT calculated or inferred.** It comes directly from the `gross_concession` column in the PSE export.

`gross_concession` = the amount Amazon refunded to the customer because of this issue (damage, non-delivery, etc.).

- A package can have a concession regardless of whether PS was effective
- A package can be effective (associate fixed it correctly) but the customer still got a refund
- This tool simply shows the raw numbers grouped by category — no inferences are made
""")

    with st.expander("💰 Refunds by Category", expanded=True):
        cdf = df[df["Cost (£)"]>0]
        if len(cdf)>0:
            cc = cdf.groupby("Category").agg(Events=("Scannable ID","count"), Refund=("Cost (£)","sum")).sort_values("Refund",ascending=False).reset_index()
            cc["Avg/Event"]=(cc["Refund"]/cc["Events"]).apply(fmt_cost); cc["Refund"]=cc["Refund"].apply(fmt_cost)
            cc.index=range(1,len(cc)+1); st.dataframe(cc, use_container_width=True)
        else: st.info("No refund data.")

    with st.expander("💰 Refunds by Process"):
        cp = df[df["Cost (£)"]>0].groupby("Process").agg(Events=("Scannable ID","count"),Refund=("Cost (£)","sum")).sort_values("Refund",ascending=False).reset_index()
        if len(cp)>0:
            cp["Refund"]=cp["Refund"].apply(fmt_cost); cp.index=range(1,len(cp)+1)
            st.dataframe(cp, use_container_width=True)

    with st.expander("🎯 DEA Misses — By Shift", expanded=True):
        st.caption("DEA Miss = package not dispatched on time. Source: `dea_miss` column in PSE export.")
        dea_ev = df[df["DEA Miss"]>0]
        if len(dea_ev)>0:
            st.error(f"🚨 {len(dea_ev)} DEA miss event(s)")
            dea_shift = dea_ev[dea_ev["Shift"].isin(SHIFT_ORDER)].groupby("Shift").size().reindex(SHIFT_ORDER,fill_value=0)
            st.pyplot(make_bar_shift(dea_shift, "DEA Misses by Shift"))
            for s in SHIFT_ORDER:
                se = dea_ev[dea_ev["Shift"]==s]
                if len(se)>0:
                    with st.expander(f"  {s} — {len(se)} miss(es) ({SHIFT_DEFINITIONS[s]})", expanded=False):
                        cols=[c for c in ["Scannable ID","Process","Category","PS Display","dea_bucket","Status"] if c in se.columns]
                        st.dataframe(se[cols].reset_index(drop=True), use_container_width=True)
        else: st.success("✅ No DEA misses.")


def render_holes(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    st.markdown("### 🕳️ Holes in Problem Solve")
    st.caption("Systemic gaps — what's failing, what keeps coming back, and who needs support.")

    # ─── BIGGEST GAPS (shown directly, no expander) ─────────────────────────
    st.markdown("#### 📊 Biggest Gaps — Categories with Worst Effectiveness")
    cat_eff = df.groupby("Category").agg(Total=("Scannable ID","count"), Effective=("Is Effective","sum"))
    cat_eff["Ineffective"] = cat_eff["Total"] - cat_eff["Effective"]
    cat_eff["Eff %"] = (cat_eff["Effective"]/cat_eff["Total"]*100).round(1)
    cat_eff = cat_eff.sort_values("Eff %", ascending=True)
    st.dataframe(cat_eff[["Total","Ineffective","Eff %"]], use_container_width=True)
    worst_cat = cat_eff[cat_eff["Total"]>=5].head(3)
    if len(worst_cat)>0:
        st.markdown("**Focus areas (5+ events, lowest Eff%):**")
        for name, row in worst_cat.iterrows():
            st.markdown(f"- 🔴 **{name}**: {int(row['Ineffective'])} ineffective out of {int(row['Total'])} ({row['Eff %']}% effective)")

    st.markdown("---")

    # ─── REPEAT PACKAGES ────────────────────────────────────────────────────
    st.markdown("#### 🔁 Packages Problem-Solved More Than Once")
    id_counts = df.groupby("Scannable ID").size()
    repeats = id_counts[id_counts>1].sort_values(ascending=False)

    if len(repeats)==0:
        st.success("✅ No packages needed PS more than once.")
    else:
        st.error(f"🚨 **{len(repeats)} packages** needed PS more than once.")
        rdf = df[df["Scannable ID"].isin(repeats.index)].sort_values(["Scannable ID","Exception Open DT"])

        # Build comparison properly using iloc
        comps = []
        for tid in repeats.index:
            evs = rdf[rdf["Scannable ID"]==tid].reset_index(drop=True)
            if len(evs)<2: continue
            effs = [evs.iloc[0]["Effective"], evs.iloc[1]["Effective"]]
            pattern = " → ".join(["✓" if e=="Y" else "✗" for e in effs])
            same = evs.iloc[0]["PS Display"]==evs.iloc[1]["PS Display"]
            comps.append({
                "Tracking ID":tid, "Pattern":pattern, "Same Associate":("⚠️ Yes" if same else "No"),
                "1st Category":evs.iloc[0]["Category"], "1st Associate":evs.iloc[0]["PS Display"],
                "2nd Category":evs.iloc[1]["Category"], "2nd Associate":evs.iloc[1]["PS Display"],
            })
        comp_df = pd.DataFrame(comps)

        # Pattern summary
        st.markdown("""
| Pattern | Meaning | Concern |
|---------|---------|---------|
| ✗ → ✗ | Failed BOTH times — nobody fixed it | 🔴 Critical |
| ✓ → ✓ | Marked fixed twice — why back? | 🟠 Suspicious |
| ✓ → ✗ | Fixed once, came back | 🟡 Investigate |
| ✗ → ✓ | Failed first, fixed second | 🟢 OK |
""")
        pc = comp_df["Pattern"].value_counts()
        for pat, cnt in pc.items():
            same_n = comp_df[(comp_df["Pattern"]==pat)&(comp_df["Same Associate"]=="⚠️ Yes")].shape[0]
            st.markdown(f"- **{pat}**: {cnt} packages ({same_n} same associate)")

        # Full table of all repeat packages (no expander — just show it)
        st.markdown("**All repeat packages:**")
        display_cols = ["Tracking ID","Pattern","Same Associate","1st Category","1st Associate","2nd Category","2nd Associate"]
        st.dataframe(comp_df[display_cols].reset_index(drop=True), use_container_width=True, height=min(350, 35*len(comp_df)+40))

        st.markdown("---")

        # ─── PER ASSOCIATE: Why their packages came back ─────────────────────
        st.markdown("#### 👤 Per Associate — Why Their Packages Came Back")
        st.caption("Shows each associate whose first resolution didn't hold, and the reason the package was re-inducted. Source: 1st PS event per tracking ID + 2nd event category.")

        # Build per-associate reasons table
        ps_reasons = []
        for ps_name in comp_df["1st Associate"].value_counts().index:
            theirs = comp_df[comp_df["1st Associate"]==ps_name]
            reasons = theirs["2nd Category"].value_counts()
            for reason, count in reasons.items():
                ps_reasons.append({"Associate": ps_name, "Packages Came Back": count, "Came Back For": reason})

        if ps_reasons:
            ps_reasons_df = pd.DataFrame(ps_reasons).sort_values("Packages Came Back", ascending=False)
            ps_reasons_df.index = range(1, len(ps_reasons_df)+1)
            st.dataframe(ps_reasons_df, use_container_width=True)
        else:
            st.info("No repeat data to analyse.")

    st.markdown("---")

    # ─── INEFFECTIVE + CUSTOMER REFUND ───────────────────────────────────────
    st.markdown("#### 📉 Ineffective PS with Customer Refund")
    st.caption("Events where PS was ineffective AND a customer refund was issued. Source: `Effective (Y/N) = N` AND `gross_concession > 0`.")
    bad = df[(~df["Is Effective"]) & (df["Cost (£)"]>0)].sort_values("Cost (£)", ascending=False)
    if len(bad)>0:
        st.error(f"{len(bad)} events — {fmt_cost(bad['Cost (£)'].sum())} total refunds on failed PS")
        cols = [c for c in ["Scannable ID","Process","Category","PS Display","Shift","Cost (£)"] if c in bad.columns]
        st.dataframe(bad[cols].head(20).reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No ineffective events with customer refunds.")


def render_trend(df, total, dr, kp=""):
    if total==0: st.warning("No data."); return
    st.markdown("### 📈 Week-over-Week Trend")
    st.markdown("""
**How this works:** You compare your PSE numbers across multiple weeks to see if things improve.

- **Option 1 — Type numbers:** Check PSE Dashboard each week, note total + effective, type below
- **Option 2 — Upload CSVs:** Export PSE raw data once per week, upload each file here

You need **2+ weeks minimum**. 4+ is ideal for seeing a real trend.
""")
    st.markdown("---")
    tm = st.selectbox("Input method:", ["📝 Type numbers", "📂 Upload CSVs"], key=f"{kp}tm")
    if "Type" in tm:
        nw = st.slider("Weeks:", 2, 12, 4, key=f"{kp}tw_n"); weeks=[]
        for i in range(nw):
            with st.expander(f"Week {i+1}", expanded=(i<3)):
                wl=st.text_input("Label:",value=f"W{i+1}",key=f"{kp}tw_l{i}")
                wt=st.number_input("Total events:",min_value=0,value=0,step=1,key=f"{kp}tw_t{i}")
                we=st.number_input("Effective:",min_value=0,value=0,step=1,key=f"{kp}tw_e{i}")
                if wt>0: weeks.append({"Week":wl,"Total":int(wt),"Effective":int(we)})
        _trend(weeks, kp)
    else:
        nf = st.slider("Weeks:", 2, 12, 4, key=f"{kp}tf_n"); weeks=[]
        for i in range(nf):
            c1,c2=st.columns([1,3])
            with c1: wl=st.text_input("Label:",value=f"W{i+1}",key=f"{kp}tf_l{i}")
            with c2: fu=st.file_uploader("PSE CSV:",type="csv",key=f"{kp}tf_f{i}")
            if fu:
                try:
                    wdf=pd.read_csv(fu,encoding="utf-8-sig")
                    we=int((wdf.get("Effective (Y/N)",pd.Series(dtype=str)).astype(str).str.strip().str.upper()=="Y").sum()) if "Effective (Y/N)" in wdf.columns else 0
                    weeks.append({"Week":wl,"Total":len(wdf),"Effective":we})
                    st.caption(f"✓ {wl}: {len(wdf)} events, {we} effective ({fmt_pct(we,len(wdf))})")
                except Exception as e: st.error(f"Error: {e}")
        _trend(weeks, kp)

def _trend(weeks, kp=""):
    if len(weeks)>=2:
        w=pd.DataFrame(weeks); w["Ineffective"]=w["Total"]-w["Effective"]; w["Eff %"]=(w["Effective"]/w["Total"]*100).round(1)
        fig,ax=plt.subplots(figsize=(7,3))
        ax.plot(w["Week"],w["Total"],marker="o",color="steelblue",linewidth=2,label="Total")
        ax.plot(w["Week"],w["Ineffective"],marker="s",color="#e74c3c",linewidth=1.5,label="Ineffective")
        for _,r in w.iterrows(): ax.annotate(str(int(r["Total"])),xy=(r["Week"],r["Total"]),xytext=(0,8),textcoords="offset points",ha="center",fontsize=7,color="steelblue")
        ax.set_xlabel("Week"); ax.set_ylabel("Events"); ax.set_title("Weekly Trend",fontsize=9)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig)
        fig2,ax2=plt.subplots(figsize=(7,2.5))
        ax2.plot(w["Week"],w["Eff %"],marker="o",color="darkgreen",linewidth=2)
        for _,r in w.iterrows(): ax2.annotate(f"{r['Eff %']}%",xy=(r["Week"],r["Eff %"]),xytext=(0,8),textcoords="offset points",ha="center",fontsize=7)
        ax2.axhline(y=w["Eff %"].mean(),color="gray",linestyle="--",linewidth=1)
        ax2.set_xlabel("Week"); ax2.set_ylabel("Eff %"); ax2.set_title("Effectiveness Trend",fontsize=9)
        ax2.tick_params(labelsize=7); plt.xticks(rotation=45); plt.tight_layout(); st.pyplot(fig2)
        f,l = w.iloc[0]["Eff %"], w.iloc[-1]["Eff %"]
        if l>f+5: st.success(f"📈 Improving: {f}% → {l}%")
        elif l<f-5: st.error(f"📉 Worsening: {f}% → {l}%")
        else: st.info(f"➡️ Stable: {f}% → {l}%")
        st.dataframe(w, use_container_width=True)
    elif len(weeks)==1: st.info("Need 2+ weeks.")

def render_export(df, total, dr, kp=""):
    st.markdown("#### 💾 Export")
    ex=[c for c in df.columns if c.endswith("_DT") or c=="_k"]
    st.download_button("⬇️ Download data (CSV)", df[[c for c in df.columns if c not in ex]].to_csv(index=False), "PSE_Data.csv", "text/csv", key=f"{kp}dl")

# ═══════════════════════════════════════════════════════════════════════════════
# GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
def render_guide():
    st.markdown("### 📖 How to Use")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("""
**Step 1:** PSE Dashboard → Raw Data → Export CSV

**Step 2:** Upload here

**Step 3 (optional, for location data):** Summary tab → copy IDs → upload into SCC with all options selected → export CSV → upload here as SCC file
""")
    with st.expander("📊 Tabs"):
        st.markdown("""
| Tab | Shows |
|-----|-------|
| 📊 **Summary** | Overview + IDs for SCC |
| 📍 **Locations** | Cluster deep dive (needs SCC) |
| 👤 **Associates** | Ranked worst→best + flagged |
| 🔄 **Cycles** | By dispatch cycle |
| 💰 **Cost & DEA** | Customer refunds + DEA misses |
| 🕳️ **Holes** | Packages PS'd multiple times |
| 📈 **Trend** | Week-over-week |
| 💾 **Export** | Download |
""")
    with st.expander("❓ Terms"):
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Effective** | Associate fixed the problem (from `Effective (Y/N)` column) |
| **Ineffective** | Attempted but didn't fix it |
| **SLA** | Fixed within the allowed time window |
| **DEA Miss** | Package not dispatched on time |
| **Concession/Refund** | Money refunded to customer (`gross_concession` column) |
""")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def run_station(df, total, dr, kp=""):
    """Run all tabs for a given dataset."""
    t1,t2,t3,t4,t5,t6,t7,t8 = st.tabs(["📊 Summary","📍 Locations","👤 Associates","🔄 Cycles","💰 Cost & DEA","🕳️ Holes","📈 Trend","💾 Export"])
    with t1: render_summary(df, total, dr, kp)
    with t2: render_locations(df, total, dr, kp)
    with t3: render_associates(df, total, dr, kp)
    with t4: render_cycles(df, total, dr, kp)
    with t5: render_cost(df, total, dr, kp)
    with t6: render_holes(df, total, dr, kp)
    with t7: render_trend(df, total, dr, kp)
    with t8: render_export(df, total, dr, kp)

def load_dataset(pse_file, scc_file=None, label=""):
    """Load, clean, validate a PSE+SCC pair. Returns (df, error_msg)."""
    try: pse_df = pd.read_csv(pse_file, encoding="utf-8-sig")
    except Exception as e: return None, f"Error reading PSE: {e}"
    miss = [c for c in REQUIRED_PSE_COLS if c not in pse_df.columns]
    if miss: return None, f"Missing columns: {miss}"
    pse_df, removed = filter_uk_ids(pse_df)
    df = clean_pse(pse_df)
    if scc_file is not None:
        try:
            scc_df = pd.read_csv(scc_file, encoding="utf-8-sig")
            df = merge_pse_scc(df, scc_df)
        except: pass
    for col in ["Cluster","Aisle","Sort Zone"]:
        if col not in df.columns: df[col] = None
    return df, None

mode = st.radio("Mode:", ["📖 Guide", "Single Station", "Multi-Station Compare"], horizontal=True, key="mode")

if mode == "📖 Guide":
    render_guide()

elif mode == "Single Station":
    c1,c2 = st.columns(2)
    with c1: pf = st.file_uploader("🔧 PSE Dashboard CSV", type="csv", key="pse")
    with c2: sf = st.file_uploader("📋 SCC CSV (optional)", type="csv", key="scc")

    if pf:
        df, err = load_dataset(pf, sf)
        if err: st.error(f"❌ {err}"); st.stop()
        total = len(df)
        if total==0: st.warning("No events."); st.stop()
        dr = get_date_range(df)

        # Filters
        st.markdown("---")
        f1,f2 = st.columns(2)
        with f1:
            procs = sorted(df["Process"].dropna().unique().tolist())
            sel_p = st.multiselect("Process:", procs, default=procs, key="fp")
        with f2:
            cats = sorted(df["Category"].dropna().unique().tolist())
            sel_c = st.multiselect("Category:", cats, default=cats, key="fc")
        filtered = df[df["Process"].isin(sel_p) & df["Category"].isin(sel_c)] if sel_p and sel_c else df
        total = len(filtered)
        if total==0: st.warning("No events match."); st.stop()

        # Metrics
        st.markdown("---")
        e=int(filtered["Is Effective"].sum()); ie=total-e; sla=int(filtered["SLA Met"].sum()); cost=filtered["Cost (£)"].sum()
        if dr: st.caption(f"📅 **{dr}** | {total} events")
        sc,col,lab,reas = compute_health(filtered, total)
        st.markdown(f"**Health: {col} {sc}/10 — {lab}**"+(f" ({', '.join(reas)})" if reas else ""))
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Events",total); m2.metric("Effective",f"{e} ({fmt_pct(e,total)})"); m3.metric("Ineffective",f"{ie} ({fmt_pct(ie,total)})")
        m4.metric("SLA Met",fmt_pct(sla,total)); m5.metric("Refunds",fmt_cost(cost))
        st.markdown("---")
        run_station(filtered, total, dr)
    else:
        st.info("👆 Upload PSE Dashboard CSV.")

elif mode == "Multi-Station Compare":
    st.caption("Upload PSE + SCC for each station to compare side by side. All tabs work for each station.")
    num = st.slider("How many stations/datasets?", 2, 5, 2, key="ms_n")
    datasets = {}
    for i in range(num):
        with st.expander(f"Dataset {i+1}", expanded=(i<2)):
            nm = st.text_input("Label:", placeholder=f"e.g. DRM2 W28", key=f"ms_nm{i}")
            c1,c2 = st.columns(2)
            with c1: pf = st.file_uploader(f"PSE CSV", type="csv", key=f"ms_pse{i}")
            with c2: sf = st.file_uploader(f"SCC CSV (optional)", type="csv", key=f"ms_scc{i}")
            if pf:
                df, err = load_dataset(pf, sf)
                if err: st.error(err)
                elif df is not None:
                    label = nm.strip() if nm and nm.strip() else f"Dataset {i+1}"
                    datasets[label] = df

    if len(datasets) >= 2:
        st.success(f"✅ Loaded: {', '.join(datasets.keys())}")

        # Show health per station
        for name, sdf in datasets.items():
            sc,col,lab,_ = compute_health(sdf, len(sdf))
            e = int(sdf["Is Effective"].sum())
            st.caption(f"{name}: {col} {sc}/10 — {len(sdf)} events, {fmt_pct(e,len(sdf))} effective")

        # Filters (applied to all)
        st.markdown("---")
        all_procs = sorted(set().union(*[set(d["Process"].dropna().unique()) for d in datasets.values()]))
        all_cats = sorted(set().union(*[set(d["Category"].dropna().unique()) for d in datasets.values()]))
        f1,f2 = st.columns(2)
        with f1: sel_p = st.multiselect("Process:", all_procs, default=all_procs, key="ms_fp")
        with f2: sel_c = st.multiselect("Category:", all_cats, default=all_cats, key="ms_fc")

        # Apply filters
        filtered_datasets = {}
        for name, sdf in datasets.items():
            f = sdf[sdf["Process"].isin(sel_p) & sdf["Category"].isin(sel_c)] if sel_p and sel_c else sdf
            if len(f)>0: filtered_datasets[name] = f

        if len(filtered_datasets) < 2:
            st.warning("Need 2+ datasets with data after filtering."); st.stop()

        # Station selector + tabs
        st.markdown("---")
        names = list(filtered_datasets.keys())
        view = st.radio("View:", ["📊 Side-by-Side Summary", "🔍 Deep Dive (one station)"], horizontal=True, key="ms_view")

        if view == "📊 Side-by-Side Summary":
            st.markdown("### 📊 Comparison")
            # Build comparison table
            rows = []
            for name, sdf in filtered_datasets.items():
                e=int(sdf["Is Effective"].sum()); ie=len(sdf)-e; sla=int(sdf["SLA Met"].sum())
                rows.append({"Station":name, "Events":len(sdf), "Effective":e, "Ineffective":ie,
                            "Eff %":round(e/len(sdf)*100,1), "SLA %":round(sla/len(sdf)*100,1),
                            "Refunds":fmt_cost(sdf["Cost (£)"].sum()), "DEA Misses":int(sdf["DEA Miss"].sum())})
            st.dataframe(pd.DataFrame(rows).set_index("Station"), use_container_width=True)

            # Comparison chart
            fig, ax = plt.subplots(figsize=(7,3))
            x = range(len(names))
            eff_rates = [round(filtered_datasets[n]["Is Effective"].sum()/len(filtered_datasets[n])*100,1) for n in names]
            bars = ax.bar(x, eff_rates, color=["steelblue" if r>=70 else "darkorange" if r>=50 else "firebrick" for r in eff_rates])
            ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
            ax.set_ylabel("Effectiveness %", fontsize=8); ax.set_title("Effectiveness Comparison", fontsize=9)
            ax.set_ylim(0, 105)
            for i,r in enumerate(eff_rates): ax.text(i, r+1, f"{r}%", ha="center", fontsize=8)
            ax.tick_params(labelsize=7); plt.tight_layout(); st.pyplot(fig)

            # By process comparison
            st.markdown("**By Process:**")
            proc_comp = []
            for name, sdf in filtered_datasets.items():
                for p in PROCESS_ORDER:
                    psub = sdf[sdf["Process"]==p]
                    if len(psub)>0:
                        proc_comp.append({"Station":name, "Process":p, "Events":len(psub), "Eff %":round(psub["Is Effective"].sum()/len(psub)*100,1)})
            if proc_comp:
                pcomp = pd.DataFrame(proc_comp)
                pivot = pcomp.pivot_table(index="Station", columns="Process", values="Eff %", fill_value=0)
                st.dataframe(pivot, use_container_width=True)

        else:
            sel = st.selectbox("Station:", names, key="ms_sel")
            sdf = filtered_datasets[sel]
            total = len(sdf); dr = get_date_range(sdf)
            e=int(sdf["Is Effective"].sum()); ie=total-e; sla=int(sdf["SLA Met"].sum())
            st.markdown(f"### {sel}")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Events",total); m2.metric("Effective",f"{e} ({fmt_pct(e,total)})"); m3.metric("Ineffective",ie); m4.metric("SLA",fmt_pct(sla,total))
            st.markdown("---")
            run_station(sdf, total, dr, kp=f"ms_{sel}_")

    elif len(datasets)==1:
        st.info("Upload at least 2 datasets to compare.")
    else:
        st.info("👆 Upload PSE files above.")
