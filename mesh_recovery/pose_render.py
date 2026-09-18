import os
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['OMP_NUM_THREADS']='4'
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
