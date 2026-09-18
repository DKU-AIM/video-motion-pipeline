"""Export continuous single-person SMPL joint sequences, without inventing missing frames."""
import os,sys,json,pathlib,hashlib
import numpy as np
import torch
R=pathlib.Path(os.environ['MOTION_WORKSPACE']).expanduser().resolve()
sys.path.insert(0,str(R/'ml-comotion/src'))
from comotion_demo.utils.smpl_kinematics import SMPLKinematics
kin=SMPLKinematics().cpu().eval()
out=R/'motion_inputs';out.mkdir(exist_ok=True)
rows=[]
paths=list((R/'existing_results').glob('*/soccer/comotion/motion_tracks.pt'))+list((R/'existing_results').glob('*/concert/comotion/motion_tracks.pt'))
paths+=list((R/'results').glob('*/*/comotion/motion_tracks.pt')) if (R/'results').exists() else []
for p in paths:
 if not p.with_name('metadata.json').exists():continue
 m=json.loads(p.with_name('metadata.json').read_text());d=torch.load(p,map_location='cpu',weights_only=False)
 fps=m.get('fps',30.0);ids=np.asarray(d['id']);frames=np.asarray(d['frame_idx'])
 candidates=[]
 for tid in np.unique(ids):
  ix=np.where(ids==tid)[0];ix=ix[np.argsort(frames[ix])]
  cuts=np.r_[0,np.where(np.diff(frames[ix])!=1)[0]+1,len(ix)]
  for a,b in zip(cuts[:-1],cuts[1:]):
   seg=ix[a:b]
   if len(seg)/fps>=2.0:candidates.append(seg)
 candidates.sort(key=len,reverse=True)
 for rank,ix in enumerate(candidates[:2]):
  # Up to six seconds, no extrapolation or ID stitching.
  ix=ix[:int(fps*6)];tid=int(ids[ix[0]])
  with torch.inference_mode():
   j=kin(d['betas'][ix].float(),d['pose'][ix].float(),d['trans'][ix].float(),output_format='joints').numpy()[:,:22]
  assert np.isfinite(j).all() and j.shape[1:]==(22,3)
  # Camera axes x right/y down/z forward -> right-handed y-up coordinates.
  j=j*np.array([1.,-1.,-1.],dtype=np.float32)
  t=np.arange(len(j))/fps;target=np.arange(0,t[-1]+1e-8,1/20)
  res=np.stack([np.interp(target,t,j[:,k,c]) for k in range(22) for c in range(3)],axis=1).reshape(-1,22,3).astype(np.float32)
  name='_'.join(p.relative_to(R).parts[:-2])+f'_id{tid}_part{rank}'
  dst=out/(name+'.npy');np.save(dst,res)
  rows.append(dict(id=name,path=str(dst),source_parameters=str(p),track_id=tid,source_fps=fps,target_fps=20,source_frame_start=int(frames[ix[0]]),source_frame_end=int(frames[ix[-1]]),frames=len(res),duration=len(res)/20,coordinate_note='camera-relative trajectory, rotated to y-up; not world-calibrated',selection='longest contiguous tracks, no ID stitching; visual review required',sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))

if not rows:
 raise RuntimeError("No continuous motion tracks exported; check pose results before captioning.")
