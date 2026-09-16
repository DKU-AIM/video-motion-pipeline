import os
os.environ['PYOPENGL_PLATFORM']='osmesa'
os.environ['LP_NUM_THREADS']='2'
import sys,pathlib,json,numpy as np,torch
from PIL import Image
import pose_render as mod
out=pathlib.Path(sys.argv[1]);torch.set_num_threads(2)
mod.state=lambda **kw:(out/'render_status.json').write_text(json.dumps(kw))
c=torch.load(out/'render_cache.pt',map_location='cpu',weights_only=False)
from fractions import Fraction
native_paths=[pathlib.Path(p) for p in c['paths']]
stride=max(1,round(float(Fraction(c['fps']))/15))
indices=list(range(0,len(native_paths),stride))
paths=[native_paths[i] for i in indices]
display_fps=str(Fraction(c['fps'])/stride)
(out/'render_metadata.json').write_text(json.dumps(dict(native_frames=len(native_paths),display_frames=len(paths),native_fps=c['fps'],display_fps=display_fps,stride=stride,note='Only visualization is sampled; model inference and saved motion parameters retain every input frame.')))

def getframe(i,p):
 i=indices[i]
 if c['mode']=='comotion':
  ix=c['by'].get(i,[]);return np.asarray(Image.open(p).convert('RGB')),[c['meshes'][k] for k in ix],[c['faces']]*len(ix),[int(c['ids'][k]) for k in ix],c['K'].astype(float)
 r=c['frames'][i];im=mod.denormalize_rgb(mod.preprocess_image(str(p))[0].cpu().numpy());return im,r['vertices'],[c['faces']]*len(r['vertices']),r['ids'].tolist(),r['K'].astype(float)
mod.render(out,paths,display_fps,getframe,c['model'])
(out/'render_cache.pt').unlink();(out/'error.json').unlink(missing_ok=True)
print('RENDER_VERIFIED',out,flush=True)
