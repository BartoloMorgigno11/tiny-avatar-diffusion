"""CPU-only H7 reporting: snapshot plotting and strict frozen-artifact checks."""
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
SEEDS=(2026,2027,2028)
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def value(x): return x["value"] if isinstance(x,dict) else x
def strict(ca16,h7):
    ca16,h7=Path(ca16),Path(h7); rows=[]; refs=[]
    r11=ca16/"results/r11_checkpoint_selection_reproducible_tokenfix_batched_v2"
    for s in SEEDS:
        selection=read(r11/f"selections/ca16_ablation_ca8_ca16_seed{s}.json")
        base=next(x for x in selection["candidates"] if x["selected"])
        hp=h7/"results/h7_ca16_40k_v1"; h=read(hp/f"r11_terminal_seed{s}/H7_R11_COMPLETE.json");k=read(hp/f"r13_kid_seed{s}/H7_KID_COMPLETE.json");f=read(hp/f"r13_fid_seed{s}/H7_FID_COMPLETE.json");pool=read(hp/f"r13_shared_pool_b8_seed{s}/R13_POOL_COMPLETE.json")
        if any(x.get("status")!="COMPLETE" for x in (h,k,f,pool)) or h["terminal"]["seed"]!=s or k["terminal"]!=h["terminal"] or f["terminal"]!=h["terminal"]: raise SystemExit(f"invalid completion identity for seed {s}")
        a={"comp_maa":base["comp_family_macro_maa"],"comp_psr":base["comp_macro_psr"],"id_maa":base["id_maa"]}; b={"comp_maa":h["metrics"]["comp_family_macro_maa"],"comp_psr":h["metrics"]["comp_macro_psr"],"id_maa":h["metrics"]["id_maa"]}
        for prefix,z in (("kid",k),("fid",f)):
            for role in ("ID","COMP","GLOBAL"):
                a[f"{prefix}_{role.lower()}"]=value(z["baseline_20k"][f"{prefix.upper()}_{role}"] if prefix=="kid" else z["baseline_20k"]["metrics"][f"FID_{role}"])
                b[f"{prefix}_{role.lower()}"]=value(z["metrics"][f"{prefix.upper()}_{role}"] if prefix=="kid" else z["metrics"][f"FID_{role}"])
        rows.append({"seed":s,"terminal":h["terminal"],"metrics_20k":a,"metrics_40k":b,"diagnostics_40k":{"id_psr":h["metrics"]["id_metrics"]["psr"],"family_maa":{q:h["metrics"]["comp_per_family"][q]["maa"] for q in ("CV1","CV2","CV3","CV4")},"historical_20k_id_psr":None,"historical_20k_family_maa":None},"pool":pool,"marker_hashes":{"r11":sha(hp/f"r11_terminal_seed{s}/H7_R11_COMPLETE.json"),"kid":sha(hp/f"r13_kid_seed{s}/H7_KID_COMPLETE.json"),"fid":sha(hp/f"r13_fid_seed{s}/H7_FID_COMPLETE.json")}})
    return {"schema":"h7_results_snapshot_v1","software":{"matplotlib":"3.11.2","clean_fid":"0.1.35","inception_sha256":"f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4"},"protocols":{"r11_tag":"h7-terminal-protocol-v3","r11_commit":"c2eb836b2bf53bcc830e52719a996c69f9eb678c","r13_tag":"h7-r13-batched-v1","r13_commit":"39e69336ca3997971cd35fcc86ff4a3c792e1d59"},"rows":rows,"limitations":{"historical_20k_id_psr":"unavailable","historical_20k_family_maa":"unavailable"}}
def plot(snap,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); rows=snap["rows"]; colors={2026:"#31688e",2027:"#35b779",2028:"#440154"}; markers={2026:"o",2027:"s",2028:"^"}
    def save(fig,n): fig.tight_layout();fig.savefig(out/f"{n}.png",dpi=220);fig.savefig(out/f"{n}.pdf");plt.close(fig)
    def pair(key,title,ylabel,n,zero=False):
        fig,ax=plt.subplots(figsize=(7,4.2))
        for i,r in enumerate(rows):
            y=[r["metrics_20k"][key],r["metrics_40k"][key]];ax.plot([0,1],y,color=colors[r["seed"]],marker=markers[r["seed"]],label=f"seed {r['seed']}");ax.annotate(f"{r['seed']}",(1,y[1]),xytext=(5,0),textcoords="offset points",fontsize=8)
        ax.set(xlim=(-.15,1.25),xticks=[0,1],xticklabels=["20k","40k"],xlabel="Training step",ylabel=ylabel,title=title);ax.set_ylim(bottom=0 if zero else 0);ax.grid(axis="y",alpha=.25);ax.legend(title="Trajectory");save(fig,n)
    pair("comp_maa","COMP_VAL equal-family macro MAA (higher is better)","MAA","01_comp_maa");pair("comp_psr","COMP_VAL MacroPSR (higher is better)","PSR","02_comp_macro_psr")
    for prefix,title,n in (("kid","KID (lower is better)","03_kid"),("fid","FID (secondary; lower is better)","04_fid")):
        fig,axs=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
        for ax,role in zip(axs,("id","comp","global")):
            for r in rows: ax.plot([0,1],[r['metrics_20k'][f'{prefix}_{role}'],r['metrics_40k'][f'{prefix}_{role}']],color=colors[r['seed']],marker=markers[r['seed']],label=f"seed {r['seed']}")
            ax.set(ylim=(0,None),xticks=[0,1],xticklabels=["20k","40k"],title=role.upper());ax.grid(axis="y",alpha=.25)
        axs[0].set_ylabel(prefix.upper());axs[0].legend(title="Seed");fig.suptitle(title);save(fig,n)
    fig,ax=plt.subplots(figsize=(7,4.2))
    for r in rows: ax.plot(["CV1","CV2","CV3","CV4"],[r['diagnostics_40k']['family_maa'][q] for q in ("CV1","CV2","CV3","CV4")],color=colors[r['seed']],marker=markers[r['seed']],label=f"seed {r['seed']}")
    ax.set(ylim=(0,1),ylabel="40k MAA (higher is better)",title="COMP_VAL family diagnostics (40k only)");ax.grid(axis="y",alpha=.25);ax.legend();save(fig,"05_family_maa_40k")
def main():
 p=argparse.ArgumentParser();p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--figures",type=Path,required=True);p.add_argument("--verify",action="store_true");p.add_argument("--ca16-root",type=Path);p.add_argument("--h7-root",type=Path);p.add_argument("--write-snapshot",type=Path);a=p.parse_args()
 if a.verify:
  if not(a.ca16_root and a.h7_root):raise SystemExit("--verify requires --ca16-root and --h7-root")
  actual=strict(a.ca16_root,a.h7_root); expected=read(a.snapshot) if a.snapshot.exists() else None
  if expected and actual!=expected: raise SystemExit("strict verification failed: snapshot differs from frozen artifacts")
  if a.write_snapshot: dump(a.write_snapshot,actual)
  snap=actual
 else:snap=read(a.snapshot)
 plot(snap,a.figures)
if __name__=="__main__":main()
