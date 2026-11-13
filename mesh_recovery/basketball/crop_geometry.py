import math

def expand_box(box, width, height, margin=0.1):
    x,y,w,h=map(float,box)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<=0:
        raise ValueError('invalid bbox')
    rect=(max(0,math.floor(x-margin*w)),max(0,math.floor(y-margin*h)),
          min(width,math.ceil(x+(1+margin)*w)),min(height,math.ceil(y+(1+margin)*h)))
    if rect[2]<=rect[0] or rect[3]<=rect[1]: raise ValueError('bbox outside image')
    return rect

def segment_boxes(frames, ident, start, end):
    lookup={f['frame']:f for f in frames}
    boxes=[]
    for i in range(start,end):
        matches=[o['bbox_xywh'] for o in lookup.get(i,{}).get('observations',[]) if o['id']==ident]
        if len(matches)!=1: raise ValueError(f'missing or duplicate target ID {ident} at frame {i}')
        boxes.append(matches[0])
    return boxes
