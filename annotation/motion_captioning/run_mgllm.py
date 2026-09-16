import os
import pathlib,sys,json,time,zipfile
import numpy as np,torch
from transformers import T5Tokenizer,T5ForConditionalGeneration
R=pathlib.Path(os.environ['MOTION_WORKSPACE']).expanduser().resolve();repo=R/'MG-MotionLLM';sys.path.insert(0,str(repo))
from options import option
from models.vqvae import HumanVQVAE
with zipfile.ZipFile(R/'assets/t2m.zip') as z:
 for suffix in ['mean.npy','std.npy']:
  name='t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/'+suffix;z.extract(name,repo/'checkpoints')
meta=repo/'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta';mean=np.load(meta/'mean.npy');std=np.load(meta/'std.npy')
args=option.get_args_parser().parse_args([]);args.nb_joints=22
torch.set_num_threads(2);torch.manual_seed(0)
vae=HumanVQVAE(args,512,args.code_dim,args.output_emb_width,2,args.stride_t,args.width,3,args.dilation_growth_rate)
vae.load_state_dict(torch.load(repo/'checkpoints/pretrained_vqvae/t2m.pth',map_location='cpu',weights_only=False)['net'],strict=True);vae.cuda().eval()
rows=json.loads((R/'results_motiongpt/captions.json').read_text());out=R/'results_mgllm';out.mkdir(exist_ok=True);allresults=[]
for task in ['m2t','m2dt']:
 tokenizer=T5Tokenizer.from_pretrained(R/'assets'/('mgllm_'+task));model=T5ForConditionalGeneration.from_pretrained(R/'assets'/('mgllm_'+task)).cuda().eval()
 for row in rows:
  feats=np.load(R/'results_motiongpt'/row['id']/'features_263.npy');x=torch.from_numpy((feats-mean)/std).float().cuda()[None]
  with torch.inference_mode():
   codes=vae.encode(x).cpu().numpy()[0].reshape(-1).tolist();motion='<Motion Tokens>'+''.join('<'+str(t)+'>' for t in codes)+'</Motion Tokens>'
   instruction='Generate text: ' if task=='m2t' else 'Generate the motion script: '
   ids=tokenizer(instruction+motion,return_tensors='pt').input_ids.cuda();t=time.time()
   generated=model.generate(ids,max_length=40 if task=='m2t' else 1536,num_beams=1,do_sample=False)
   text=tokenizer.decode(generated[0],skip_special_tokens=True).strip('"')
  result=dict(id=row['id'],task=task,model='wbz0505/'+task+'-ft-from-GSPretrained-base',caption=text,seconds=time.time()-t,input_modalities=['3D motion only'],prompt=instruction+'<motion tokens>',feature_frames=len(feats),tokens=len(codes))
  allresults.append(result);(out/'captions.json').write_text(json.dumps(allresults,ensure_ascii=False,indent=2));print(task,row['id'],text,flush=True)
 del model;torch.cuda.empty_cache()
(out/'status.json').write_text(json.dumps({'state':'complete','samples':len(allresults)}))
