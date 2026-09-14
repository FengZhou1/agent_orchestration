"""Fit a low-dimensional average waiting approximation using long-horizon runs."""

from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from agent_orch.validation.llmservingsim import read_simulator_output
RES=ROOT/"results"
OUT=RES/"llm_queue_waiting_fit"; OUT.mkdir(parents=True,exist_ok=True)

# Long-run saturated throughput: use the largest measured value across the
# 2x/4x/8x probes.
cap_frames=[]
for d,n in ((RES/"llm_queue_capacity_long","capacity_long_observations.csv"),(RES/"llm_queue_capacity_long8","capacity_long8_observations.csv")):
    f=d/n
    if f.exists(): cap_frames.append(pd.read_csv(f))
cap=pd.concat(cap_frames,ignore_index=True).groupby("config_id").measured_throughput_rps.max().to_dict()

rows=[]
def add_run(root:Path, run_id:str, config_id:str, load_factor:float, arrival_rate:float, source:str):
    sim=read_simulator_output(root/"runs"/f"{run_id}.csv").sort_values("end_s")
    manifest=pd.read_csv(root/"manifests"/f"{run_id}.csv")
    merged=sim.merge(manifest,on="request_id",validate="one_to_one")
    sample=merged.iloc[int(.1*len(merged)):int(.9*len(merged))]
    ends=sim.end_s.values; lo=int(.1*len(ends)); hi=int(.9*len(ends)); thr=(hi-lo)/(ends[hi]-ends[lo])
    rows.append({"config_id":config_id,"source":source,"load_factor":load_factor,"arrival_rate_rps":arrival_rate,"capacity_rps":cap[config_id],"rho":arrival_rate/cap[config_id],"observed_waiting_s":float(sample.waiting_s.mean()),"observed_ttft_s":float(sample.ttft_s.mean()),"observed_response_s":float(sample.response_s.mean()),"observed_service_s":float(sample.service_s.mean()),"measured_throughput_rps":thr,"n":len(sample)})

# Long-horizon stable runs: recover the per-run arrival rate from the manifest
# by using the calibrated old capacity and the requested load factor.
oldcap=pd.read_csv(RES/"llm_queue_trends"/"calibration.csv").set_index("config_id").saturated_capacity_rps.to_dict()
for run_id in sorted((RES/"llm_queue_longstable"/"runs").glob("*.csv")):
    stem=run_id.stem
    # Names are longstable-<config>-load<value>.
    prefix="longstable-"; body=stem[len(prefix):]; load_text=body.rsplit("-load",1)[1]; config_id=body.rsplit("-load",1)[0]
    load=float(load_text)
    # The longstable script used the long-run capacity, not the old capacity.
    add_run(RES/"llm_queue_longstable",stem,config_id,load,load*cap[config_id],"long_stable")

# Short overload probes.
over=RES/"llm_queue_overload"/"overload_observations.csv"
if over.exists():
    for r in pd.read_csv(over).itertuples(index=False):
        rows.append({"config_id":r.config_id,"source":"overload_short","load_factor":r.load_factor,"arrival_rate_rps":r.arrival_rate_rps,"capacity_rps":cap[r.config_id],"rho":r.arrival_rate_rps/cap[r.config_id],"observed_waiting_s":r.observed_waiting_s,"observed_ttft_s":r.observed_ttft_s,"observed_response_s":r.observed_response_s,"observed_service_s":r.observed_service_s,"measured_throughput_rps":np.nan,"n":getattr(r,"n_requests",getattr(r,"n",-1))})
long=RES/"llm_queue_overload_long"/"long_overload_observations.csv"
if long.exists():
    for r in pd.read_csv(long).itertuples(index=False):
        rows.append({"config_id":r.config_id,"source":"overload_long","load_factor":r.load_factor,"arrival_rate_rps":r.arrival_rate_rps,"capacity_rps":cap[r.config_id],"rho":r.arrival_rate_rps/cap[r.config_id],"observed_waiting_s":r.observed_waiting_s,"observed_ttft_s":r.observed_ttft_s,"observed_response_s":r.observed_response_s,"observed_service_s":r.observed_service_s,"measured_throughput_rps":np.nan,"n":getattr(r,"n_requests",getattr(r,"n",-1))})

data=pd.DataFrame(rows)
data.to_csv(OUT/"waiting_observations_all.csv",index=False)
stable=data[(data.source=="long_stable") & (data.rho<1.0)].copy().sort_values(["config_id","rho"])
stable["log_wait"]=np.log(stable.observed_waiting_s.clip(lower=1e-12))
configs=sorted(stable.config_id.unique()); n=len(configs)

def fit_power(per_a=True, per_w0=True, global_beta=True):
    def unpack(p):
        i=0
        w0={}
        a={}
        for c in configs:
            w0[c]=np.exp(p[i]) if per_w0 else 0.0; i+=1
            a[c]=np.exp(p[i]); i+=1
        beta=p[i] if global_beta else None
        return w0,a,beta
    def pred(p):
        w0,a,beta=unpack(p); out=[]
        for c in configs:
            d=stable[stable.config_id==c]; rho=d.rho.values
            if global_beta: b=beta
            else: b=1.0
            out.extend(np.log(w0[c]+a[c]*np.power(rho/(1-rho),b)))
        return np.array(out)
    k=2*n+(1 if global_beta else 0); p0=np.zeros(k); p0[0::2]=np.log(0.01); p0[1::2]=np.log(0.1)
    if global_beta: p0[-1]=0.3
    res=least_squares(lambda p:pred(p)-stable.log_wait.values,p0,max_nfev=400000)
    pr=np.exp(pred(res.x)); return res,pr,k
res,pr,k=fit_power(); stable["pred_power"]=pr
metrics={"n":len(stable),"params":k,"log_rmse":float(np.sqrt(np.mean((np.log(pr)-stable.log_wait.values)**2))),"median_ratio":float(np.median(pr/stable.observed_waiting_s)),"spearman":float(pd.Series(pr).corr(stable.observed_waiting_s,method="spearman"))}
print("power fit",metrics)
print(stable[["config_id","source","rho","observed_waiting_s","pred_power"]].to_string(index=False))
(OUT/"waiting_model_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")