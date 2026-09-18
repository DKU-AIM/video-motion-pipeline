import os
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['OMP_NUM_THREADS']='4'
import argparse
parser=argparse.ArgumentParser(description='Run mesh recovery for prepared clips.')
parser.add_argument('--models',nargs='+',choices=['comotion','hmr2'],default=['comotion','hmr2'])
parser.add_argument('--conditions',nargs='+',choices=['full','crop'],help='Clip conditions to run. Defaults to full/crop, except daily uses full only.')
args=parser.parse_args()

import sys,pathlib,json,time,subprocess,pickle,traceback
import numpy as np
import torch
from PIL import Image,ImageDraw
R=pathlib.Path(os.environ['MOTION_WORKSPACE']).expanduser().resolve();sys.path.insert(0,str(R))
from smpl_eval.meshrender import MeshRenderer,track_color,draw_labels
from smpl_eval.overlay import _font
from comotion_demo.utils import smpl_kinematics as sk
OUT=R/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
def state(**kw):(OUT/'status.json').write_text(json.dumps(dict(updated=time.time(),**kw),indent=2))
def arr(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else x
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(p)]))['streams'][0]
def render(out,paths,fps,getframe,model):
 w=640;h=None;enc=None;renderer=None
 try:
  for i,p in enumerate(paths):
   im,verts,faces,ids,K=getframe(i,p)
   ih,iw=im.shape[:2];hh=round(ih/iw*w/2)*2
   if h is None:
    h=hh;renderer=MeshRenderer(w,h)
    enc=subprocess.Popen(['ffmpeg','-y','-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w*2}x{h}','-r',fps,'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'comparison.mp4')],stdin=subprocess.PIPE)
   assert hh==h
   im=np.asarray(Image.fromarray(im).resize((w,h),Image.Resampling.LANCZOS));K=K.copy();K[0]*=w/iw;K[1]*=h/ih
   focal=(K[0,0],K[1,1]);center=(K[0,2],K[1,2])
   ov=renderer.render(im,verts,faces,focal,center,[track_color(x) for x in ids]);ov=draw_labels(ov,list(zip(ids,verts)),focal,center,12)
   canvas=Image.fromarray(np.concatenate([im,ov],axis=1));d=ImageDraw.Draw(canvas)
   for x,label in [(0,'Input (display resized)'),(w,model+' | '+out.parent.name)]:
    d.rectangle((x,0,x+w,24),fill=(15,20,28));d.text((x+7,2),label,font=_font(16),fill='white')
   enc.stdin.write(canvas.tobytes())
   if i in (0,len(paths)//2,len(paths)-1):canvas.save(out/f'qa_{i:05d}.jpg')
   if i%60==0:state(state='render',case=str(out.relative_to(OUT)),frame=i,total=len(paths))
 finally:
  if renderer:renderer.close()
  if enc:enc.stdin.close();assert enc.wait()==0
 p=probe(out/'comparison.mp4');assert int(p['nb_frames'])==len(paths)
 subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(out/'comparison.mp4'),'-f','null','-'],check=True)
 (out/'verified.json').write_text(json.dumps(dict(frames=len(paths),duration=p['duration'],decode='passed'),indent=2))
render_jobs=[]
def launch_render(out,paths,fps,model,payload):
 global render_jobs
 while len([p for p in render_jobs if p.poll() is None])>=2:time.sleep(2)
 payload.update(paths=[str(p) for p in paths],fps=fps,model=model)
 torch.save(payload,out/'render_cache.pt')
 log=open(out/'render.log','w')
 job=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).with_name('render_cache.py')),str(out)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','LP_NUM_THREADS':'2'})
 render_jobs.append(job);log.close()

sess=None;fail=[]
rows=json.loads((R/'clip_manifest.json').read_text())
# Full and crop remain separate experiments. Daily's ROI equals full image; no duplicate experiment.
for row in rows:
 default_conditions=['full','crop'] if row['id']!='daily' else ['full']
 for condition in args.conditions or default_conditions:
  src=R/'inputs'/row['id']/(condition+'.mp4');pr=probe(src);fps=pr['avg_frame_rate'];fn,fd=map(int,fps.split('/'));hz=fn/fd;N=int(pr['nb_frames']);W,H=pr['width'],pr['height']
  for model in args.models:
   out=OUT/row['id']/condition/model;out.mkdir(parents=True,exist_ok=True)
   if (out/'verified.json').exists():continue
   try:
    state(state='inference',case=str(out.relative_to(OUT)))
    frames=out/'frames';frames.mkdir(exist_ok=True)
    if model=='comotion':
     if len(list(frames.glob('*.png')))!=N:
      subprocess.run(['ffmpeg','-y','-v','error','-threads','2','-i',str(src),'-vsync','0',str(frames/'%06d.png')],check=True)
     paths=sorted(frames.glob('*.png'));assert len(paths)==N
     raw=out/'raw';raw.mkdir(exist_ok=True);t=time.time()
     cmd=[sys.executable,str(R/'ml-comotion/demo.py'),'-i',str(frames),'-o',str(raw),'--skip-visualization','--frameskip','1','--num-frames',str(N)]
     if not (out/'motion_tracks.pt').exists():
      with open(out/'inference.log','w') as log:subprocess.run(cmd,cwd=R/'ml-comotion',stdout=log,stderr=subprocess.STDOUT,check=True)
      d=torch.load(raw/'frames.pt',map_location='cpu',weights_only=False);torch.save(d,out/'motion_tracks.pt')
      (out/'runtime.json').write_text(json.dumps({'inference_seconds':time.time()-t}))
     d=torch.load(out/'motion_tracks.pt',map_location='cpu',weights_only=False)
     for key in ['pose','trans','betas']:assert torch.isfinite(d[key]).all()
     assert d['pose'].shape[-1]==72
     with open(sk.smpl_model_path,'rb') as f:faces=np.asarray(pickle.load(f,encoding='latin1')['f'],np.int32)
     kin=sk.SMPLKinematics().cpu().eval();meshes=[]
     with torch.inference_mode():
      for a in range(0,len(d['pose']),128):meshes.append(kin(d['betas'][a:a+128],d['pose'][a:a+128],d['trans'][a:a+128],output_format='mesh').numpy())
     meshes=np.concatenate(meshes) if meshes else np.empty((0,6890,3))
     by={}
     for i,fr in enumerate(d['frame_idx'].tolist()):by.setdefault(fr,[]).append(i)
     counts=[len(by.get(f,[])) for f in range(N)]
     K=np.array([[2*max(W,H),0,W/2],[0,2*max(W,H),H/2],[0,0,1]],dtype=float)
     def getframe(i,p):
      ix=by.get(i,[]);return np.asarray(Image.open(p).convert('RGB')),[meshes[k] for k in ix],[faces]*len(ix),[int(d['id'][k]) for k in ix],K
     meta=dict(body_model='SMPL neutral',track_ids=sorted(set(map(int,d['id']))),K=K.tolist(),frameskip=1)
    else:
     from multihmr2 import api
     from multihmr2.datasets.itw_image import preprocess_image
     from multihmr2.utils import denormalize_rgb
     if sess is None:sess=api.init_hmr_session(str(R/'multi-hmr2/checkpoints/multihmr2.pt'))
     t=time.time();preds=api.infer_video(sess,str(src),str(frames),conf_thresh=.4,dist_thresh_nms=.25,lowres=False);torch.cuda.synchronize();runtime=time.time()-t
     paths=sorted(frames.glob('*.png'))
     if len(paths)==N+1 and len(preds)==N+1 and np.array_equal(np.asarray(Image.open(paths[-1])),np.asarray(Image.open(paths[-2]))):paths=paths[:-1];preds=preds[:-1]
     assert len(preds)==len(paths)==N,(len(preds),len(paths),N)
     records=[];counts=[]
     for i,pred in enumerate(preds):
      p=pred.persons;pp=p.bone_poses.clone().float();pp[:,[0],:3,-1]=p.transl_pelvis.unsqueeze(1)
      rec=dict(frame_index=i,track_id=arr(p.track_id),shape=arr(p.shape),pose_parameters=arr(pp),transl_pelvis=arr(p.transl_pelvis),joints3d_camera=arr(p.j3d),K_processed=arr(pred.K),confidence=arr(p.conf))
      for key in ['shape','pose_parameters','joints3d_camera']:assert np.isfinite(rec[key]).all()
      records.append(rec);counts.append(int(p.num_person))
     torch.save({'frames':records},out/'motion_tracks.pt');(out/'runtime.json').write_text(json.dumps({'inference_seconds':runtime}))
     faces=arr(sess.model.full_body_decoder.body_model.faces)
     def getframe(i,p):
      pred=preds[i];im=denormalize_rgb(preprocess_image(str(p))[0].cpu().numpy());verts=[arr(v).reshape(-1,3) for v in pred.persons.v3d];return im,verts,[faces]*len(verts),arr(pred.persons.track_id).tolist(),arr(pred.K).astype(float)
     meta=dict(body_model='Anny',track_ids=sorted(set(int(x) for f in records for x in f['track_id'])),conf_thresh=.4,dist_thresh_nms=.25,lowres=False)
    meta.update(model=model,source=str(src),source_start_seconds=row['source_start_seconds'],category=row['category'],crop=condition=='crop',roi_xywh=row['roi_xywh'] if condition=='crop' else None,source_resolution=[W,H],fps=hz,fps_fraction=fps,frames=N,persons_per_frame=counts,coordinate_system='camera-relative, uncalibrated',git_commit=subprocess.check_output(['git','-C',str(R/('ml-comotion' if model=='comotion' else 'multi-hmr2')),'rev-parse','HEAD'],text=True).strip(),**json.loads((out/'runtime.json').read_text()))
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=True,indent=2))
    if model=='comotion':
     payload=dict(mode=model,meshes=meshes,faces=faces,by=by,ids=arr(d['id']),K=K)
    else:
     payload=dict(mode=model,faces=faces,frames=[dict(vertices=[arr(v).reshape(-1,3) for v in pr.persons.v3d],ids=arr(pr.persons.track_id),K=arr(pr.K)) for pr in preds])
    launch_render(out,paths,fps,model,payload)
    print('INFERENCE_SAVED',str(out.relative_to(OUT)),flush=True)
   except Exception as e:
    traceback.print_exc();fail.append({'case':str(out.relative_to(OUT)),'error':str(e)});(out/'error.json').write_text(json.dumps(fail[-1],indent=2))
state(state='waiting_for_render',failures=fail)
for job in render_jobs:
 if job.wait()!=0:fail.append({'stage':'render','returncode':job.returncode})
state(state='complete' if not fail else 'completed_with_failures',failures=fail)
if fail: sys.exit(1)
