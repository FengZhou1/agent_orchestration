"""Long-horizon stable-load runs using long-run saturated capacity."""

from __future__ import annotations
import math, shutil, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(ROOT/"src"))
import validate_llm_queue_trends as base
from agent_orch.validation.llmservingsim import read_simulator_output
REMOTE="zf@192.168.234.128"; REMOTE_REPO="/home/zf/桌面/LLMServingSim"; REMOTE_STAGE="outputs/llm_queue_longstable"
LOADS=(0.5,0.8,0.95); REQUESTS=128; SEED=20260917
def run(cmd,timeout=None):
 p=subprocess.run(cmd,check=False,text=True,encoding="utf-8",errors="replace",stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
 if p.returncode: raise RuntimeError(p.stdout[-3000:])
 return p.stdout
def main():
 result=ROOT/"results"/"llm_queue_longstable"; staging_root=result/"staging"
 if staging_root.exists(): shutil.rmtree(staging_root); staging_root.mkdir(parents=True,exist_ok=True)
 frames=[]
 for name in ("capacity_long_observations.csv","capacity_long8_observations.csv"):
  path=ROOT/"results"/("llm_queue_capacity_long" if "8" not in name else "llm_queue_capacity_long8")/name
  if path.exists(): frames.append(pd.read_csv(path))
 capframe=pd.concat(frames,ignore_index=True); cap=capframe.groupby("config_id").measured_throughput_rps.max().to_dict()
 print("effective long-run capacity",cap)
 base.REMOTE_STAGE=REMOTE_STAGE; jobs=[]
 for cfg in base.CONFIGS:
  for load in LOADS:
   jobs.append({"job_id":base.job_id("longstable",cfg["id"],f"load{load:.2f}"),"kind":"long_stable","config_id":cfg["id"],"composition":base.COMPOSITIONS["mixed"],"composition_name":"mixed","num_requests":REQUESTS,"arrival_rate_rps":load*float(cap[cfg["id"]]),"load_factor":load,"seed":SEED,"simultaneous":False})
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
  cfg=next(c for c in base.CONFIGS if c["id"]==item["config_id"]); sim=read_simulator_output(result/"runs"/f"{item['job_id']}.csv").sort_values("end_s"); ends=sim.end_s.values; lo=int(.1*len(ends)); hi=int(.9*len(ends)); thr=(hi-lo)/(ends[hi]-ends[lo]); sample=sim.iloc[lo:hi]
  rows.append({"job_id":item["job_id"],"config_id":cfg["id"],"load_factor":item["load_factor"],"arrival_rate_rps":item["arrival_rate_rps"],"capacity_rps":cap[cfg["id"]],"observed_waiting_s":float(sample.waiting_s.mean()),"observed_ttft_s":float(sample.ttft_s.mean()),"observed_response_s":float(sample.response_s.mean()),"observed_service_s":float(sample.service_s.mean()),"measured_throughput_rps":thr,"n":len(sample)})
 frame=pd.DataFrame(rows); frame.to_csv(result/"longstable_observations.csv",index=False); print(frame.to_string(index=False))
if __name__=="__main__": main()