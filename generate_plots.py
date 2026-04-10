"""
generate_thesis_figures.py
──────────────────────────
Generuje grafy a tabuľky pre diplomovú prácu z výsledkov experimentov.
Štýl grafov inšpirovaný článkom Sarnovský & Kolařík (2021), PeerJ CS.

Vstupy (z results/ adresára):
  - grid_search_results_raw.csv
  - grid_search_summary.csv
  - prequential_block_metrics.csv

Výstupy (do results/figures/) — všetko .png:
  - performance_synth_<metric>.png     — grid: všetky modely na syntetických ds
  - performance_real_<metric>.png      — grid: všetky modely na reálnych ds
  - ddcw_all_<metric>.png              — grid: DDCW vs HT baseline na všetkých
  - showcase_<dataset>.png             — jeden dataset, viac metrík vedľa seba
  - training_times.png                 — grouped bar chart
  - summary_table.tex                  — LaTeX tabuľka s bold pre najlepšie
  - times_table.tex                    — LaTeX tabuľka časov

Spustenie:
    python generate_thesis_figures.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR = "./results/"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures/")
os.makedirs(FIGURES_DIR, exist_ok=True)

METRICS = ["RWA_Score", "G_Mean", "Mean_Minority_Recall", "Macro_F1"]

MODEL_SHORT = {
    "AdaptiveRandomForestClassifier": "ARF",
    "OzaBaggingADWINClassifier":      "OzaBagADWIN",
    "LeveragingBaggingClassifier":    "LevBagging",
    "OnlineBoostingClassifier":       "OnlineBoosting",
    "HoeffdingTreeClassifier":        "HoeffdingTree",
}

# Farby a čiary — DDCW oranžová (plná hrubá), ARF modrá, ostatné tenšie
STYLE = {
    "DDCW":          {"color": "#d35400", "lw": 2.0, "ls": "-",  "zorder": 10},
    "ARF":           {"color": "#2980b9", "lw": 1.3, "ls": "-",  "zorder": 5},
    "OzaBagADWIN":   {"color": "#27ae60", "lw": 1.1, "ls": "--", "zorder": 4},
    "LevBagging":    {"color": "#8e44ad", "lw": 1.1, "ls": "--", "zorder": 4},
    "OnlineBoosting":{"color": "#c0392b", "lw": 1.0, "ls": ":",  "zorder": 3},
    "HoeffdingTree": {"color": "#7f8c8d", "lw": 1.0, "ls": ":",  "zorder": 2},
}
_DFLT = {"color": "#34495e", "lw": 1.0, "ls": "-", "zorder": 1}

REAL_NAMES = {"ELEC", "KDD99", "Airlines", "Shuttle", "CoverType", "Jigsaw"}
DPI = 300

# Akademický font štýl
plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def sn(name):
    if name.startswith("DDCW"): return "DDCW"
    return MODEL_SHORT.get(name, name[:20])

def st(m): return STYLE.get(m, _DFLT)

def load():
    d = {}
    for k, f in [("raw","grid_search_results_raw.csv"),
                 ("summary","grid_search_summary.csv"),
                 ("blocks","prequential_block_metrics.csv")]:
        p = os.path.join(RESULTS_DIR, f)
        if os.path.exists(p):
            d[k] = pd.read_csv(p); print(f"  OK  {f:<40} {len(d[k]):>7} rows")
    return d

# ═══════════════════════════════════════════════════════════════════════════════
# SUBPLOT: priebeh metriky, všetky modely, jeden dataset
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_ds(ax, blk, ds, metric, ylabel=True):
    sub = blk[blk["Dataset"]==ds].copy()
    if sub.empty: ax.text(.5,.5,"no data",transform=ax.transAxes,ha="center"); return
    sub["M"] = sub["Model"].apply(sn)
    for m in sorted(sub["M"].unique(), key=lambda x:(0 if x=="DDCW" else 1,x)):
        g = sub[sub["M"]==m].groupby("Block_End")[metric].agg(["mean","std"]).reset_index()
        g["std"]=g["std"].fillna(0); s=st(m)
        ax.plot(g["Block_End"],g["mean"],label=m,color=s["color"],
                linewidth=s["lw"],linestyle=s["ls"],zorder=s["zorder"])
        if m=="DDCW":
            ax.axhline(g["mean"].mean(),color=s["color"],lw=.6,ls=":",alpha=.45,zorder=s["zorder"]-1)
    ax.set_title(ds,fontweight="bold",pad=4); ax.set_xlabel("Samples")
    if ylabel: ax.set_ylabel(metric.replace("_"," "))
    ax.set_ylim(bottom=max(0,ax.get_ylim()[0]))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PERFORMANCE GRID  (Figure 4/5 štýl)
# ═══════════════════════════════════════════════════════════════════════════════

def perf_grid(blk, dsets, metric, fname, prefix=""):
    v=[d for d in dsets if d in blk["Dataset"].unique()]
    if not v: return
    n=len(v); nc=2 if n<=4 else 3; nr=(n+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(nc*4.8,nr*3.2),squeeze=False)
    for i,ds in enumerate(v):
        r,c=divmod(i,nc); _plot_ds(axes[r,c],blk,ds,metric,ylabel=(c==0))
    for i in range(n,nr*nc): r,c=divmod(i,nc); axes[r,c].set_visible(False)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=min(6,len(l)),frameon=True,
               fancybox=False,edgecolor="#ccc",bbox_to_anchor=(.5,-.02))
    fig.suptitle(f"{prefix}{metric.replace('_',' ')}",fontsize=11,fontweight="bold",y=1.01)
    fig.tight_layout(rect=[0,.04,1,1.])
    fig.savefig(os.path.join(FIGURES_DIR,fname),dpi=DPI); plt.close(fig)
    print(f"  ✓  {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. DDCW vs BASELINE na všetkých streamoch
# ═══════════════════════════════════════════════════════════════════════════════

def ddcw_grid(blk, metric, fname):
    df=blk.copy(); df["M"]=df["Model"].apply(sn)
    df=df[df["M"].isin(["DDCW","HoeffdingTree"])]
    if df.empty: return
    dsets=sorted(df["Dataset"].unique()); n=len(dsets)
    nc=3 if n>4 else 2; nr=(n+nc-1)//nc
    fig,axes=plt.subplots(nr,nc,figsize=(nc*4.5,nr*3.),squeeze=False)
    for i,ds in enumerate(dsets):
        r,c=divmod(i,nc); ax=axes[r,c]; dd=df[df["Dataset"]==ds]
        for m in ["DDCW","HoeffdingTree"]:
            g=dd[dd["M"]==m]
            if g.empty: continue
            a=g.groupby("Block_End")[metric].agg(["mean","std"]).reset_index(); s=st(m)
            ax.plot(a["Block_End"],a["mean"],label=m,color=s["color"],lw=s["lw"],ls=s["ls"])
            ax.axhline(a["mean"].mean(),color=s["color"],lw=.5,ls=":",alpha=.4)
        ax.set_title(ds,fontsize=8,fontweight="bold",pad=3)
        ax.set_xlabel("Samples",fontsize=7)
        if c==0: ax.set_ylabel(metric.replace("_"," "),fontsize=7)
        ax.set_ylim(bottom=max(0,ax.get_ylim()[0]))
    for i in range(n,nr*nc): r,c=divmod(i,nc); axes[r,c].set_visible(False)
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=2,frameon=True,bbox_to_anchor=(.5,-.01))
    fig.suptitle(f"DDCW vs HoeffdingTree — {metric.replace('_',' ')}",
                 fontsize=11,fontweight="bold",y=1.01)
    fig.tight_layout(rect=[0,.04,1,1.]); fig.savefig(os.path.join(FIGURES_DIR,fname),dpi=DPI)
    plt.close(fig); print(f"  ✓  {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHOWCASE — jeden dataset, viac metrík
# ═══════════════════════════════════════════════════════════════════════════════

def showcase(blk, ds, mlist, fname):
    av=[m for m in mlist if m in blk.columns]
    if not av: return
    n=len(av); fig,axes=plt.subplots(1,n,figsize=(n*4.8,3.8))
    if n==1: axes=[axes]
    for i,m in enumerate(av): _plot_ds(axes[i],blk,ds,m,ylabel=True)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="lower center",ncol=min(6,len(l)),frameon=True,bbox_to_anchor=(.5,-.03))
    fig.suptitle(f"Porovnanie modelov — {ds}",fontsize=11,fontweight="bold",y=1.02)
    fig.tight_layout(rect=[0,.05,1,1.])
    fig.savefig(os.path.join(FIGURES_DIR,fname),dpi=DPI); plt.close(fig)
    print(f"  ✓  {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRÉNINGOVÉ ČASY
# ═══════════════════════════════════════════════════════════════════════════════

def times_chart(raw, fname):
    if "Total_Time_s" not in raw.columns: return
    df=raw.copy(); df["M"]=df["Model"].apply(sn)
    agg=df.groupby(["Dataset","M"])["Total_Time_s"].mean().reset_index()
    piv=agg.pivot(index="Dataset",columns="M",values="Total_Time_s")
    cols=sorted(piv.columns,key=lambda x:(0 if x=="DDCW" else 1,x)); piv=piv[cols]
    colors=[st(c)["color"] for c in cols]
    fig,ax=plt.subplots(figsize=(12,5))
    piv.plot(kind="bar",ax=ax,width=.75,color=colors,edgecolor="white",linewidth=.5)
    ax.set_ylabel("Priemerný čas (s)"); ax.set_title("Tréningové časy",fontweight="bold")
    ax.legend(title="Model",bbox_to_anchor=(1.02,1),loc="upper left",frameon=True)
    plt.xticks(rotation=35,ha="right"); fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR,fname),dpi=DPI); plt.close(fig)
    print(f"  ✓  {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5–6. LATEX TABUĽKY
# ═══════════════════════════════════════════════════════════════════════════════

def latex_table(summ, fname):
    df=summ.copy(); df["M"]=df["Model"].apply(sn)
    cols=["Avg_RWA","Avg_G_Mean","Avg_Macro_F1","Avg_Weighted_F1","Avg_Mean_Minority_Recall"]
    av=[c for c in cols if c in df.columns]
    if not av: return
    hdr={"Avg_RWA":"RWA","Avg_G_Mean":"G-Mean","Avg_Macro_F1":"Macro F1",
         "Avg_Weighted_F1":"W. F1","Avg_Mean_Minority_Recall":"Min. Rec."}
    L=["\\begin{table}[h]","\\centering",
       "\\caption{Porovnanie výkonnosti modelov.}","\\label{tab:results_main}","\\small",
       "\\begin{tabular}{ll"+"r"*len(av)+"}","\\toprule",
       "Dataset & Model & "+" & ".join(hdr.get(c,c) for c in av)+" \\\\","\\midrule"]
    prev=None
    for ds in sorted(df["Dataset"].unique()):
        g=df[df["Dataset"]==ds].sort_values("M")
        if prev is not None: L.append("\\midrule")
        bests={c:g[c].max() for c in av}
        for _,row in g.iterrows():
            ds_s=ds if ds!=prev else ""
            vals=[]
            for c in av:
                v=row[c]; s=f"{v:.4f}" if not pd.isna(v) else "—"
                if not pd.isna(v) and abs(v-bests[c])<1e-6: s="\\textbf{"+s+"}"
                vals.append(s)
            L.append(f"{ds_s} & {row['M']} & "+" & ".join(vals)+" \\\\"); prev=ds
    L+=["\\bottomrule","\\end{tabular}","\\end{table}"]
    p=os.path.join(FIGURES_DIR,fname)
    with open(p,"w",encoding="utf-8") as f: f.write("\n".join(L))
    print(f"  ✓  {fname}")

def latex_times(raw,fname):
    if "Total_Time_s" not in raw.columns: return
    df=raw.copy(); df["M"]=df["Model"].apply(sn)
    agg=df.groupby(["Dataset","M"])["Total_Time_s"].mean().reset_index()
    piv=agg.pivot(index="Dataset",columns="M",values="Total_Time_s")
    ms=sorted(piv.columns,key=lambda x:(0 if x=="DDCW" else 1,x))
    L=["\\begin{table}[h]","\\centering",
       "\\caption{Tréningové časy (s).}","\\label{tab:times}","\\small",
       "\\begin{tabular}{l"+"r"*len(ms)+"}","\\toprule",
       "Dataset & "+" & ".join(ms)+" \\\\","\\midrule"]
    for ds in piv.index:
        vs=[f"{piv.loc[ds,m]:.0f}" if m in piv.columns and not pd.isna(piv.loc[ds,m]) else "—" for m in ms]
        L.append(f"{ds} & "+" & ".join(vs)+" \\\\")
    L+=["\\bottomrule","\\end{tabular}","\\end{table}"]
    p=os.path.join(FIGURES_DIR,fname)
    with open(p,"w",encoding="utf-8") as f: f.write("\n".join(L))
    print(f"  ✓  {fname}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("  Generovanie grafov — štýl Sarnovský & Kolařík (2021)")
    print("="*65)
    print("\nNačítavam...")
    data=load()
    if not data: print("  Žiadne dáta."); sys.exit(1)

    if "blocks" in data:
        blk=data["blocks"]
        a=sorted(blk["Dataset"].unique())
        sy=[d for d in a if d not in REAL_NAMES]
        re=[d for d in a if d in REAL_NAMES]
        print("\n── Performance gridy ──")
        for m in METRICS:
            if m not in blk.columns: continue
            if sy: perf_grid(blk,sy,m,f"performance_synth_{m.lower()}.png","Syntetické — ")
            if re: perf_grid(blk,re,m,f"performance_real_{m.lower()}.png","Reálne — ")
        print("\n── DDCW vs baseline ──")
        for m in METRICS:
            if m in blk.columns: ddcw_grid(blk,m,f"ddcw_all_{m.lower()}.png")
        print("\n── Showcase ──")
        am=[m for m in METRICS if m in blk.columns]
        for ds in a: showcase(blk,ds,am,f"showcase_{ds.lower().replace(' ','_')}.png")

    if "raw" in data:
        print("\n── Časy ──"); times_chart(data["raw"],"training_times.png")
    if "summary" in data:
        print("\n── LaTeX tabuľky ──"); latex_table(data["summary"],"summary_table.tex")
    if "raw" in data:
        latex_times(data["raw"],"times_table.tex")
    print(f"\n{'='*65}\n  Hotovo. → {os.path.abspath(FIGURES_DIR)}\n{'='*65}")

if __name__=="__main__": main()