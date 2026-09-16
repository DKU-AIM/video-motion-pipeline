"""Official MotionGPT VQ/LM classes and m2t generation; no RGB or category text in prompt."""
import os,sys,json,time,pathlib,types,hashlib
import numpy as np
import torch
R=pathlib.Path(os.environ['MOTION_WORKSPACE']).expanduser().resolve();repo=R/'MotionGPT';sys.path.insert(0,str(repo))
# Avoid importing training dataloaders/evaluation dependencies for inference-only conversion.
for name in ['mGPT.data','mGPT.data.humanml','mGPT.data.humanml.common','mGPT.data.humanml.scripts','mGPT.data.humanml.utils']:
 mod=types.ModuleType(name);mod.__path__=[str(repo/name.replace('.','/'))];sys.modules[name]=mod
from mGPT.archs.mgpt_vq import VQVae
from mGPT.archs.mgpt_lm import MLM
from mGPT.data.humanml.scripts import motion_process as mp
from mGPT.data.humanml.common.skeleton import Skeleton
from mGPT.data.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain
torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
out=R/'results_motiongpt';out.mkdir(exist_ok=True)
state=lambda **kw:(out/'status.json').write_text(json.dumps(kw,indent=2))
state(state='loading')
ckpt=torch.load(R/'assets/motiongpt_s3_h3d.tar',map_location='cpu',weights_only=False)
sd=ckpt.get('state_dict',ckpt)
vae=VQVae(nfeats=263,down_t=2,norm=None)
vae.load_state_dict({k[4:]:v for k,v in sd.items() if k.startswith('vae.')},strict=True)
lm=MLM(model_path='google/flan-t5-base',stage='lm_instruct')
lm.load_state_dict({k[3:]:v for k,v in sd.items() if k.startswith('lm.')},strict=True)
vae=vae.cuda().eval();lm=lm.cuda().eval();del sd,ckpt
mean=np.load(repo/'assets/meta/mean.npy');std=np.load(repo/'assets/meta/std.npy')
assert mean.shape==std.shape==(263,) and np.all(std>0)
mp.l_idx1,mp.l_idx2=5,8;mp.fid_r,mp.fid_l=[8,11],[7,10];mp.face_joint_indx=[2,1,17,16];mp.joints_num=22
mp.n_raw_offsets=torch.from_numpy(t2m_raw_offsets);mp.kinematic_chain=t2m_kinematic_chain
ref=np.load(R/'assets/reference_joints.npy').astype(np.float32)
mp.tgt_offsets=Skeleton(mp.n_raw_offsets,mp.kinematic_chain,'cpu').get_offsets_joints(torch.from_numpy(ref[0]))
rows=json.loads((R/'motion_inputs/manifest.json').read_text());results=[]
# Public HumanML3D example is an inference sanity check, not an ETRI test result.
rows=[dict(id='official_reference_012314',path=str(R/'assets/reference_features.npy'),already_features=True)]+rows
for row in rows:
 state(state='inference',sample=row['id'],completed=len(results),total=len(rows))
 case=out/row['id'];case.mkdir(exist_ok=True)
 if row.get('already_features'):
  feats=np.load(row['path']).astype(np.float32);error=None
 else:
  joints=np.load(row['path']);feats,canonical,_,_=mp.process_file(joints.copy(),.002)
  rec=mp.recover_from_ric(torch.from_numpy(feats).float(),22).numpy()
  error=float(np.abs(rec-canonical[:-1]).max());assert error<.002,(row['id'],error)
  np.save(case/'joints_canonical.npy',canonical);np.save(case/'joints_recovered.npy',rec)
 feats=np.asarray(feats,dtype=np.float32);assert feats.ndim==2 and feats.shape[1]==263 and np.isfinite(feats).all()
 np.save(case/'features_263.npy',feats)
 x=torch.from_numpy((feats-mean)/std).float().cuda()[None]
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode():
  tokens,_=vae.encode(x);text=lm.generate_conditional(motion_tokens=tokens,lengths=[tokens.shape[1]],task='m2t')
 torch.cuda.synchronize()
 result={**row,'caption':text[0],'model':'OpenMotionLab/MotionGPT-base stage3','seconds':time.perf_counter()-start,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9,'feature_frames':len(feats),'tokens':tokens.shape[1],'roundtrip_max_abs_m':error,'prompt':'Generate text: <motion tokens>','input_modalities':['3D motion only'],'generation':'official generate_conditional m2t, greedy, max_length=40'}
 (case/'caption.json').write_text(json.dumps(result,indent=2));results.append(result)
 (out/'captions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(row['id'],text[0],flush=True)
state(state='complete',samples=len(results))
