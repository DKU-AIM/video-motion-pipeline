"""Bounded native-pixel player crops; fresh HMR2; render only selected PHALP ID.

The existing PHALP observations supply identities/boxes. No new tracker, identity
correction, detector, pose interpolation or smoothing is applied in this pass.
"""
from pathlib import Path
import os
import argparse,contextlib,io,inspect,json,subprocess,sys,time,hashlib
import numpy as np
for k,v in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'str':str,'unicode':str}.items():
    if k not in np.__dict__:setattr(np,k,v)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
R=Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ['AIM_4DHUMANS_DIR']).expanduser().resolve()))
import cv2,torch,pyrender,trimesh
from torch.utils.data._utils.collate import default_collate
from hmr2.models import load_hmr2,DEFAULT_CHECKPOINT
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils import recursive_to
from hmr2.utils.renderer import cam_crop_to_full,create_raymond_lights
from crop_geometry import expand_box,segment_boxes


class SurfaceRenderer:
    def __init__(self,w,h,faces,focal):
        self.w,self.h,self.faces,self.focal=w,h,faces,focal
        self.renderer=pyrender.OffscreenRenderer(w,h)
    def frame(self,image,verts,cam):
        scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.35]*3)
        scene.add(pyrender.IntrinsicsCamera(fx=self.focal,fy=self.focal,cx=self.w/2,cy=self.h/2,zfar=100000),pose=np.eye(4))
        xyz=(verts+cam)*np.array([1,-1,-1],np.float32)
        mat=pyrender.MetallicRoughnessMaterial(metallicFactor=0,roughnessFactor=.7,baseColorFactor=(.24,.72,1,1))
        scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(xyz,self.faces,process=False),material=mat,smooth=True))
        for light in create_raymond_lights():scene.add_node(light)
        color,depth=self.renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
        mask=depth>0
        if mask.sum()<20:raise RuntimeError('empty mesh projection')
        result=image.copy()
        result[mask]=(image[mask]*.15+color[:,:,:3][:,:,::-1][mask]*.85).astype(np.uint8)
        return result,mask
    def close(self):self.renderer.delete()

class Writer:
    def __init__(self,path,source,w,h,fps,start,duration):
        self.path=path;self.tmp=path.with_suffix('.partial.mp4')
        self.log=path.with_suffix('.ffmpeg.log').open('w')
        args=['ffmpeg','-hide_banner','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0',
              '-ss',f'{start:.9f}','-t',f'{duration:.9f}','-i',str(source),'-map','0:v:0','-map','1:a:0?',
              '-c:v','libx264','-threads','2','-preset','medium','-crf','16','-pix_fmt','yuv420p',
              '-c:a','aac','-b:a','192k','-t',f'{duration:.9f}','-movflags','+faststart',str(self.tmp)]
        self.p=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=self.log,stderr=self.log)
    def write(self,frame):self.p.stdin.write(frame.tobytes())
    def close(self):
        self.p.stdin.close();code=self.p.wait();self.log.close()
        if code:raise RuntimeError(f'ffmpeg failure: {self.path}')
        self.tmp.replace(self.path)

def probe_video(path,n,fps,w,h):
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
    v=next(s for s in p['streams'] if s['codec_type']=='video')
    assert int(v['nb_frames'])==n,(path,v['nb_frames'],n)
    assert (v['width'],v['height'])==(w,h)
    a,b=map(int,v['avg_frame_rate'].split('/'));assert abs(a/b-fps)<1e-8
    aud=next(s for s in p['streams'] if s['codec_type']=='audio')
    assert abs(float(aud['duration'])-n/fps)<.07
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    return {'file':path.name,'frames':n,'fps':fps,'width':w,'height':h,'audio_duration':aud['duration'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def run_segment(seg,model,cfg,smoke,source_dir,destination):
    t=time.monotonic();case=seg['case'];ident=seg['id'];start=seg['start'];end=seg['end']
    data=json.loads((source_dir/case/'observed_tracks.json').read_text())
    boxes=segment_boxes(data['frames'],ident,start,end)
    rects=[expand_box(b,data['width'],data['height']) for b in boxes]
    W=(max(r[2]-r[0] for r in rects)+1)//2*2
    H=(max(r[3]-r[1] for r in rects)+1)//2*2
    fps=data['fps'];n=end-start
    out=destination/case/('smoke' if smoke else '선수별_크롭_메쉬')/seg['name']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'PASS.json').exists(): print('SKIP_COMPLETE',str(out),flush=True);return
    source=source_dir/case/'입력_원본클립.mp4'
    cap=cv2.VideoCapture(str(source));assert (int(cap.get(3)),int(cap.get(4)))==(data['width'],data['height'])
    focal=float(cfg.EXTRA.FOCAL_LENGTH/cfg.MODEL.IMAGE_SIZE*max(W,H))
    renderer=SurfaceRenderer(W,H,model.smpl.faces,focal)
    selected=sorted(set([0,n//2,n-1, min(n-1,max(0,1250-start))])) if smoke else list(range(n))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    writers=[]
    if not smoke:
        for name,w,h in [('01_선수크롭.mp4',W,H),('02_HMR2_SMPL_메쉬.mp4',W,H),('03_크롭_메쉬_비교.mp4',W*2,H+54)]:
            writers.append(Writer(out/name,source,w,h,fps,start/fps,n/fps))
    records=[];saved={k:[] for k in ['vertices','joints3d','joints2d_canvas','camera_translation','pred_cam','global_orient','body_pose','betas']}
    review_idx=set(np.linspace(0,n-1,9,dtype=int).tolist())|{min(n-1,max(0,1250-start))}
    audit=[]
    for i in range(n):
        ok,frame=cap.read();assert ok,i
        if smoke and i not in selected:continue
        x0,y0,x1,y1=rects[i];cw,ch=x1-x0,y1-y0;px=(W-cw)//2;py=(H-ch)//2
        crop=frame[y0:y1,x0:x1].copy()
        canvas=np.zeros((H,W,3),dtype=np.uint8);canvas[py:py+ch,px:px+cw]=crop
        assert np.array_equal(canvas[py:py+ch,px:px+cw],crop)
        # Expanded crop bounds identify the target, not a fresh all-person detector.
        b=np.array([[px,py,px+cw,py+ch]],np.float32)
        with contextlib.redirect_stdout(io.StringIO()):item=ViTDetDataset(cfg,canvas,b)[0]
        batch=recursive_to(default_collate([item]),'cuda')
        with torch.inference_mode():
            pred=model(batch)
            camera=cam_crop_to_full(pred['pred_cam'],batch['box_center'].float(),batch['box_size'].float(),batch['img_size'].float(),focal)
        verts=pred['pred_vertices'][0].cpu().numpy();cam=camera[0].cpu().numpy();joints=pred['pred_keypoints_3d'][0].cpu().numpy()
        assert verts.shape==(6890,3) and np.isfinite(verts).all() and np.isfinite(cam).all()
        xyz=joints+cam
        xy=xyz[:,:2]/xyz[:,2:]*focal+np.array([W/2,H/2])
        overlay,mask=renderer.frame(canvas,verts,cam)
        # Pixel mapping from cropped canvas back into original native input.
        rec={'local_frame':len(records),'input_frame':start+i,'input_time_seconds':(start+i)/fps,
             'source_time_seconds':seg['source_offset']+(start+i)/fps,'track_id':ident,
             'bbox_xywh':boxes[i],'crop_xyxy':rects[i],'canvas_offset_xy':[px,py],
             'canvas_to_input_translation_xy':[x0-px,y0-py]}
        records.append(rec)
        for k,val in [('vertices',verts),('joints3d',joints),('joints2d_canvas',xy),('camera_translation',cam),('pred_cam',pred['pred_cam'][0].cpu().numpy())]:saved[k].append(val)
        for k in ['global_orient','body_pose','betas']:saved[k].append(pred['pred_smpl_params'][k][0].cpu().numpy())
        mesh_outside_crop=int(np.count_nonzero(mask))-int(np.count_nonzero(mask[py:py+ch,px:px+cw]))
        edge_pixels=int(mask[0,:].sum()+mask[-1,:].sum()+mask[:,0].sum()+mask[:,-1].sum())
        audit.append({'input_frame':start+i,'mesh_pixels_outside_image_crop':mesh_outside_crop,'mesh_canvas_edge_pixels':edge_pixels})
        compare=np.zeros((H+54,W*2,3),np.uint8);compare[54:,:W]=canvas;compare[54:,W:]=overlay
        cv2.putText(compare,f'ID {ident} | source {rec["source_time_seconds"]:.3f}s | input f{start+i}',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(compare,'Native crop +10% / edge',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(compare,'Fresh HMR2 / SMPL',(W+8,43),cv2.FONT_HERSHEY_SIMPLEX,.40,(100,220,255),1,cv2.LINE_AA)
        if not smoke:
            for wr,im in zip(writers,[canvas,overlay,compare]):wr.write(im)
        if smoke or i in review_idx:
            cv2.imwrite(str(out/f'review_{start+i:05d}.jpg'),compare,[cv2.IMWRITE_JPEG_QUALITY,98])
            if i==min(n-1,max(0,1250-start)) or i==n//2:
                trimesh.Trimesh(verts,model.smpl.faces,process=False).export(out/f'SMPL_frame_{start+i:05d}.obj')
        if len(records)%60==0 or smoke:
            progress={'case':case,'id':ident,'segment':seg['name'],'processed':len(records),'total':len(selected),'elapsed_seconds':round(time.monotonic()-t,1)}
            (DEST/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    cap.release();renderer.close()
    for wr in writers:wr.close()
    arrays={k:np.stack(v).astype(np.float32) for k,v in saved.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    arrays.update(faces=np.asarray(model.smpl.faces,np.int32),input_frames=np.array([r['input_frame'] for r in records]),
                  source_times=np.array([r['source_time_seconds'] for r in records]),
                  crop_xyxy=np.array([r['crop_xyxy'] for r in records]),canvas_offsets=np.array([r['canvas_offset_xy'] for r in records]),
                  fps=np.array(fps),focal_length=np.array(focal),canvas_wh=np.array([W,H]),track_id=np.array(ident))
    np.savez_compressed(out/'SMPL_전체프레임.npz',**arrays)
    (out/'crop_mapping.json').write_text(json.dumps({'coordinate_convention':'zero based, xyxy right/bottom exclusive; output canvas copied without pixel scaling', 'segment':seg,'fps':fps,'canvas_wh':[W,H],'margin_each_side':.1,'frames':records},ensure_ascii=False,indent=2))
    (out/'projection_audit.json').write_text(json.dumps(audit))
    streams=[] if smoke else [probe_video(out/name,n,fps,w,h) for name,w,h in [('01_선수크롭.mp4',W,H),('02_HMR2_SMPL_메쉬.mp4',W,H),('03_크롭_메쉬_비교.mp4',W*2,H+54)]]
    report={'case':case,'segment':seg,'smoke':smoke,'frames':len(records),'fps':fps,'canvas_wh':[W,H],
        'crop_width_range':[min(r[2]-r[0] for r in rects),max(r[2]-r[0] for r in rects)],
        'crop_height_range':[min(r[3]-r[1] for r in rects),max(r[3]-r[1] for r in rects)],
        'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':time.monotonic()-t,
        'body_model':'SMPL','mesh_inference':'Fresh HMR 2.0 inference from selected player crop +10% per edge; not cached whole-frame SMPL',
        'input_source':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'checkpoint':DEFAULT_CHECKPOINT,'temporal_smoothing':False,'identity_interpolation':False,'player_pixels_upscaled':False,
        'input_model_tensor':list(batch['img'].shape),'backbone_tensor':'B,3,256,192 (author HMR2 normalization and aspect transform)',
        'focal_length':focal,'camera_note':'Estimated per-person camera in moving crop frame; not calibrated world-space trajectory.',
        'frames_mesh_outside_crop':sum(r['mesh_pixels_outside_image_crop']>0 for r in audit),
        'frames_mesh_touching_canvas_edge':sum(r['mesh_canvas_edge_pixels']>0 for r in audit),
        'audit_note':'Projected inferred surface bounds are a diagnostic, not ground truth for visible hand/ball coverage.',
        'streams':streams,'quality_note':'Execution checks, not pose accuracy or dense identity GT. Rectangular crop can retain other people. Target is selected by PHALP box; model may err under occlusion. Audio trimmed and encoded AAC192k; native FPS; videos H264 CRF16.'}
    (out/('SMOKE.json' if smoke else 'PASS.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2));print('SEGMENT_PASS',case,seg['name'],len(records),flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--segment', type=int)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plan', type=Path, default=R/'plan.json')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    model,cfg=load_hmr2();model=model.cuda().eval()
    (args.output_dir/'model_config.yaml').write_text(cfg.dump())
    plan=json.loads(args.plan.read_text());segments=plan['segments']
    if args.segment is not None:segments=[segments[args.segment]]
    for seg in segments:run_segment(seg,model,cfg,args.smoke,args.input_dir,args.output_dir)
    print('BATCH_COMPLETE',flush=True)

if __name__=='__main__':main()
