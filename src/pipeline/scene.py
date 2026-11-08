"""Scene object aggregation and group-level artifact filters."""
from collections import defaultdict

import numpy as np

GEOMETRIC_THRESHOLD = .70


def box_overlap(boxes,others,*,union=False):
    boxes = np.asarray(boxes,dtype=float).reshape(-1,4)
    others = np.asarray(others,dtype=float).reshape(-1,4)
    intersection = np.maximum(0,np.minimum(boxes[:,None,2:],others[None,:,2:])
                              -np.maximum(boxes[:,None,:2],others[None,:,:2])).prod(axis=2)
    area = (boxes[:,2:]-boxes[:,:2]).prod(axis=1)[:,None]
    if union:
        area = area+(others[:,2:]-others[:,:2]).prod(axis=1)[None,:]-intersection
    return intersection/np.maximum(area,1)


def persistent_regions(samples, *, tiles, image_area, threshold=.25):
    groups = defaultdict(list)
    for n,sample in enumerate(samples):
        for obj in sorted(sample['objects'],key=lambda row:-row['score']):
            kind = obj['kind']
            if kind=='person' or obj['score']<threshold or (not tiles and obj['tile']!='full'):
                continue
            box = obj['box']
            # Very large predictions are usually entire walls and rooms.
            if (box[2]-box[0])*(box[3]-box[1])>.32*image_area:
                continue
            clusters = groups[kind]
            if clusters:
                overlap = box_overlap([box],[c['median'] for c in clusters],union=True)[0]
                j = int(overlap.argmax())
                if overlap[j]>=.43:
                    cluster = clusters[j]
                    if n not in cluster['samples']:
                        cluster['samples'][n] = obj
                        cluster['median'] = np.median([o['box'] for o in cluster['samples'].values()],axis=0)
                    continue
            clusters.append({'median':box,'samples':{n:obj}})
    return [
        {'kind':kind,'box':np.asarray(c['median']).tolist(),'support':len(c['samples']),
         'median_score':float(np.median([o['score'] for o in c['samples'].values()])),
         'label':max(c['samples'].values(),key=lambda o:o['score'])['label']}
        for kind,clusters in groups.items() for c in clusters if len(c['samples'])>=3
    ]


def grouped_boxes(tracks,assignment):
    rows = defaultdict(list)
    for local,observations in tracks.items():
        rows[assignment[local]].extend(observations)
    return rows


def static_descriptor(rows):
    if len(rows)<6 or len({x[0] for x in rows})!=1:
        return None
    boxes = [row[3] for row in rows]
    median = np.median(boxes,axis=0)
    if np.min(box_overlap(boxes,[median],union=True))<.89:
        return None
    return rows[0][0],min(r[2] for r in rows),max(r[2] for r in rows),median


def recurring_static(groups):
    descriptors = [(g,d) for g,rows in groups.items() if (d:=static_descriptor(rows)) is not None]
    flagged = set()
    for i,(left,(cam,a,b,box)) in enumerate(descriptors):
        for right,(other,c,d,other_box) in descriptors[i+1:]:
            if cam==other and max(c-b,a-d)>=45 and box_overlap([box],[other_box],union=True)[0,0]>=.91:
                flagged.update((left,right))
    return flagged


def covered_groups(groups,regions,kinds):
    flagged = set()
    for group,rows in groups.items():
        by_camera = defaultdict(list)
        for camera,_,_,box in rows:
            by_camera[camera].append(box)
        matches = True
        for camera,boxes in by_camera.items():
            allowed = [o for o in regions.get(camera,[]) if o['kind'] in kinds]
            furniture = {'seat','table'}
            if static_descriptor(rows) is None:
                allowed = [o for o in allowed if o['kind'] not in furniture]
            if not allowed or np.any(box_overlap(boxes,[o['box'] for o in allowed]).max(axis=1)<GEOMETRIC_THRESHOLD):
                matches = False
                break
        if matches:
            flagged.add(group)
    return flagged


def mirror_vote_groups(groups, votes):
    flagged = set()
    for group, rows in groups.items():
        camera = rows[0][0]
        if any(row[0] != camera for row in rows):
            continue
        boxes = [row[3] for row in rows]
        support = np.zeros(len(boxes), dtype=int)
        for candidates in votes.get(camera, ()):
            if candidates:
                support += box_overlap(boxes, candidates).max(axis=1) >= .70
        if np.all(support >= 4):
            flagged.add(group)
    return flagged


def sample_frames(rows):
    # Build the camera model from its opening three minutes only.
    targets = np.linspace(rows[0]['pts'] + 3, rows[0]['pts'] + 170, 7)
    selected = {}
    for target in targets:
        row = min(rows, key=lambda row: abs(row['pts'] - target))
        selected[row['frame_index']] = row
    return list(selected.values())


def regions(image, tiles):
    w,h = image.size
    yield 'full',(0,0,w,h)
    if tiles:
        yield 'top_left',(0,0,round(.62*w),round(.66*h))
        yield 'top_right',(round(.38*w),0,w,round(.66*h))
        yield 'bottom_left',(0,round(.34*h),round(.62*w),h)
        yield 'bottom_right',(round(.38*w),round(.34*h),w,h)


COCO_KINDS = {'tv': 'screen', 'chair': 'seat', 'couch': 'seat',
              'bench': 'seat', 'dining table': 'table'}
DINO_PROMPTS = {
    'television': 'screen', 'computer monitor': 'screen', 'wall mirror': 'mirror',
    'full length mirror': 'mirror', 'glass partition': 'glass', 'office chair': 'seat',
    'armchair': 'seat', 'stool': 'seat', 'foosball table': 'table', 'dining table': 'table',
    'person': 'person',
}


def dino_detector(checkpoint, device):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, local_files_only=True)
    model = model.eval().to(device)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)

    @torch.inference_mode()
    def predict(image):
        inputs = processor(images=image, text=list(DINO_PROMPTS), return_tensors='pt',
                           size={'shortest_edge': 576, 'longest_edge': 832}).to(device)
        result = processor.post_process_grounded_object_detection(
            model(**inputs), inputs.input_ids, threshold=.17, text_threshold=.19,
            target_sizes=[(image.height, image.width)])[0]
        return [(DINO_PROMPTS[label], label, box, score)
                for label, box, score in zip(result['text_labels'], result['boxes'].tolist(),
                                             result['scores'].tolist(), strict=True)
                if label in DINO_PROMPTS]

    return predict


def discover_scene(cached, videos, yolo, model_dir, device, progress, *, need_coco, need_mirrors):
    from .vision import frames_at_pts, local_model

    missing = sorted(set(cached) - videos.keys())
    if missing:
        raise ValueError(f'Scene-dependent profile requires scene calibration or video for cameras {missing}')
    dino = dino_detector(local_model(model_dir, 'grounding-dino-tiny'), device) if need_mirrors else None
    coco_classes = [index for index, label in yolo.names.items() if label in COCO_KINDS] if need_coco else []

    def coco(image):
        result = yolo.predict(image, device=device, imgsz=960, conf=.16,
                              classes=coco_classes, verbose=False)[0]
        return [(COCO_KINDS[label], label, box, score)
                for index, box, score in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist(),
                                             result.boxes.conf.tolist(), strict=True)
                if (label := yolo.names[int(index)]) in COCO_KINDS]

    stable, votes = {}, {}
    for camera, data in cached.items():
        progress(f'Camera {camera}: discovering persistent scene objects')
        height, width = data['config']['image_size']
        image_area = height * width
        camera_samples = sample_frames(data['frames']) if data['frames'] else []
        images = dict(frames_at_pts(videos[camera], [sample['pts'] for sample in camera_samples]))
        proposals = {'coco': [], 'dino': []}
        for i, sample in enumerate(camera_samples):
            image = images[sample['pts']]
            for model, predict in (('coco', coco if need_coco else None), ('dino', dino)):
                if predict is None:
                    continue
                objects = []
                for tile, (x, y, right, bottom) in regions(image, i in (1, 4, 6)):
                    for kind, label, box, score in predict(image.crop((x, y, right, bottom))):
                        x1, y1, x2, y2 = np.clip(box, 0, [right - x, bottom - y] * 2)
                        if x2 > x1 and y2 > y1:
                            objects.append({'kind': kind, 'label': label, 'score': score, 'tile': tile,
                                            'box': [x1 + x, y1 + y, x2 + x, y2 + y]})
                proposals[model].append({'objects': objects})
        stable[camera] = persistent_regions(proposals['coco'], tiles=True, image_area=image_area)

        def is_mirror(obj):
            x1, y1, x2, y2 = obj['box']
            return obj['kind'] == 'mirror' and (x2 - x1) * (y2 - y1) < .12 * image_area

        stable[camera] += [obj for obj in persistent_regions(proposals['dino'], tiles=True,
                                                            threshold=.19, image_area=image_area)
                           if is_mirror(obj) and obj['support'] >= 4]
        votes[camera] = [[obj['box'] for obj in sample['objects']
                          if is_mirror(obj) and obj['score'] >= .19]
                         for sample in proposals['dino']]
    return stable, votes
