"""Long-horizon overload probe to separate finite-batch effects from backlog growth."""

from __future__ import annotations
import json, math, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
import validate_llm_queue_trends as base
from agent_orch.validation.llmservingsim import read_simulator_output
REMOTE="zf@192.168.234.128"; REMOTE_REPO="/home/zf/桌面/LLMServingSim"; REMOTE_STAGE="outputs/llm_queue_overload_long"
CONFIGS=["qwen3-4b-a10","qwen3-4b-h20","qwen3-32b-h20"]
LOADS=(1.5,3.0); REQUESTS=160; SEED=20260914
def run(cmd,timeout=None):
    p=subprocess.run(cmd,check=False,text=True,encoding="utf-8",errors="replace",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode: raise RuntimeError(p.stdout[-3000:])
    return p.stdout
def main():
    result=ROOT/"results"/"llm_queue_overload_long"; staging_root=result/"staging"
    if staging_root.exists(): shutil.rmtree(staging_root); staging_root.mkdir(parents=True,exist_ok=True)
    cal=pd.read_csv(ROOT/"results"/"llm_queue_trends"/"calibration.csv"); cap=dict(zip(cal.config_id,cal.saturated_capacity_rps,strict=True))
    base.REMOTE_STAGE=REMOTE_STAGE
    jobs=[]
    for cid in CONFIGS:
        cfg=next(c for c in base.CONFIGS if c["id"]==cid)
        for load in LOADS:
            jobs.append({"job_id":base.job_id("overlong",cid,f"load{load:.2f}"),"kind":"overload_long","config_id":cid,"composition":base.COMPOSITIONS["mixed"],"composition_name":"mixed","num_requests":REQUESTS,"arrival_rate_rps":load*float(cap[cid]),"load_factor":load,"seed":SEED,"simultaneous":False})
    staging=staging_root/REMOTE_STAGE; staging.mkdir(parents=True,exist_ok=True); base.write_jobs(staging,jobs)
    for sub in ("configs","workloads","manifests"):
        target=result/sub
        if target.exists(): shutil.rmtree(target)
        shutil.copytree(staging/sub,target)
    run(["ssh",REMOTE,"mkdir","-p",f"{REMOTE_REPO}/outputs"]); run(["scp","-q","-r",str(staging),f"{REMOTE}:{REMOTE_REPO}/outputs/"])
    run(["ssh",REMOTE,"bash",f"{REMOTE_REPO}/{REMOTE_STAGE}/run_jobs.sh"],timeout=14400)
    local=result/"runs"; local.mkdir(parents=True,exist_ok=True); run(["scp","-q","-r",f"{REMOTE}:{REMOTE_REPO}/{REMOTE_STAGE}/runs/.",str(local)])
    rows=[]
    for item in jobs:
        cfg=next(c for c in base.CONFIGS if c["id"]==item["config_id"]); manifest=pd.read_csv(result/"manifests"/f"{item['job_id']}.csv"); sim=read_simulator_output(result/"runs"/f"{item['job_id']}.csv"); merged=sim.merge(manifest,on="request_id",validate="one_to_one").sort_values("end_s"); sample=merged.iloc[int(.1*len(merged)):int(.9*len(merged))]
        rows.append({"job_id":item["job_id"],"config_id":cfg["id"],"load_factor":item["load_factor"],"arrival_rate_rps":item["arrival_rate_rps"],"capacity_rps":cap[cfg["id"]],"observed_waiting_s":float(sample.waiting_s.mean()),"observed_ttft_s":float(sample.ttft_s.mean()),"observed_response_s":float(sample.response_s.mean()),"observed_service_s":float(sample.service_s.mean()),"n":len(sample)})
    frame=pd.DataFrame(rows); frame.to_csv(result/"long_overload_observations.csv",index=False); print(frame.to_string(index=False))
if __name__=="__main__": main()