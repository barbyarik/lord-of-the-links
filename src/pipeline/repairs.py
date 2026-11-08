"""Optional identity repairs from the MTMC association pipeline."""
from collections import defaultdict
from itertools import combinations, pairwise
from statistics import mean

import cv2
import numpy as np

from .graph import MAX_GAP, add_conflicts, collect_tracks, join

MIN_IOU = .50
MIN_CONTAINMENT = .90
MIN_COMMON_FRAMES = 3
MAX_COMMON_FRAMES = 12
MAX_SHORT_DURATION = 6.
MIN_SUPPORT_RATIO = 2
MIN_PROTOTYPE = .88
MIN_PAIR_SIMILARITY = .88
MIN_FRACTION_NEAR = .75


def iou(a,b):
    ab = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    area_a = (a[2:]-a[:2]).prod()
    area_b = (b[2:]-b[:2]).prod()
    return float(ab/max(area_a+area_b-ab,1.))


def endpoint(rows,pts,side):
    anchor = rows[0 if side=='start' else -1]
    stamp = pts[anchor[:2]]
    box = np.asarray(anchor[3])
    selected = rows[:5] if side=='start' else rows[-5:]
    times = np.asarray([pts[row[:2]]-stamp for row in selected])
    good = (np.abs(times)<=.8)&(np.abs(times)>.08)
    shifts = np.asarray([row[3] for row in selected])-box
    delta = shifts[:,:2] + (shifts[:,2:]-shifts[:,:2])/2
    velocity = np.median(delta[good]/times[good,None],axis=0) if good.any() else np.zeros(2)
    return stamp,box,np.tile(velocity,2)


def candidates(tracks,raw_edges,frames):
    pts = {(r['camera'],r['frame_index']):r['pts'] for r in frames}
    starts = {i:endpoint(rows,pts,'start') for i,rows in tracks.items()}
    ends = {i:endpoint(rows,pts,'end') for i,rows in tracks.items()}
    choices = []
    for appearance,a,b,_ in raw_edges:
        if tracks[a][0][0]!=tracks[b][0][0]:
            continue
        a,b = sorted((a,b),key=lambda i:(starts[i][0],i))
        t1,box_a,v_a = ends[a]
        t2,box_b,v_b = starts[b]
        gap = t2-t1
        if not .001 < gap <= 1.61:
            continue
        fwd = iou(box_a+v_a*gap,box_b)
        bwd = iou(box_b-v_b*gap,box_a)
        static = iou(box_a,box_b)
        motion = .5*(max(fwd,static)+max(bwd,static))
        score = .65*motion+.35*appearance
        choices.append((score,a,b,gap,motion,appearance,static,fwd,bwd))
    return choices


def reciprocal(choices,params):
    max_gap,min_motion,min_appearance,min_margin = params
    by_source,by_target = defaultdict(list),defaultdict(list)
    for row in choices:
        score,a,b,gap,motion,appearance,*_ = row
        if gap <= max_gap+1e-4 and motion >= min_motion and appearance >= min_appearance:
            by_source[a].append(row)
            by_target[b].append(row)
    for values in (*by_source.values(),*by_target.values()):
        values.sort(key=lambda row:(-row[0],row[1],row[2]))
    accepted = []
    for options in by_source.values():
        row = options[0]
        score,a,b,*_ = row
        other = by_target[b]
        if other[0]!=row:
            continue
        second = max(options[1][0] if len(options)>1 else 0,
                     other[1][0] if len(other)>1 else 0)
        if score-second >= min_margin:
            accepted.append(row)
    return sorted(accepted,key=lambda row:(-row[0],row[1],row[2]))


def containment(a,b):
    overlap = np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(overlap / max(1.,min((a[2:]-a[:2]).prod(),(b[2:]-b[:2]).prod())))


def deduplicate(tracks, edges, frames, assignment, raw):
    appearance = {(a,b):sim for sim,a,b,_ in raw}
    occupants = defaultdict(dict)
    members = defaultdict(set)
    confidence = {(t['id'],f['frame_index']):t['score'] for f in frames for t in f['tracks']}
    for local,group in assignment.items():
        members[group].add(local)
        occupants[group].update({row[:2]:local for row in tracks[local]})

    options = []
    for score,a,b,_ in edges:
        if score<.25 or appearance[a,b]<.85:
            continue
        first,second = sorted((a,b),key=lambda t:(tracks[t][0][2],t))
        left,right = tracks[first][-1],tracks[second][0]
        if left[:2]!=right[:2] or len(tracks[first])<3 or len(tracks[second])<3:
            continue
        left_box,right_box = np.asarray(left[3]),np.asarray(right[3])
        if iou(left_box,right_box)<.50 or containment(left_box,right_box)<.90:
            continue
        options.append((score,first,second,left[:2]))
    skipped,applied = set(),[]
    assignment = dict(assignment)
    for score,first,second,key in sorted(options,reverse=True):
        left,right = sorted((assignment[first],assignment[second]))
        if left==right or occupants[left].keys() & occupants[right].keys() != {key}:
            continue
        ids = [occupants[left][key],occupants[right][key]]
        drop = min(ids,key=lambda local:(confidence[local,key[1]],local))
        skipped.add((drop,key[1]))
        for local in members[right]:
            assignment[local] = left
        members[left].update(members.pop(right))
        survivor = next(i for i in ids if i!=drop)
        occupants[left].update(occupants.pop(right))
        occupants[left][key] = survivor
        applied.append(dict(left=first,right=second,camera=key[0],frame_index=key[1],
                            score=score,dropped_local=drop))
    filtered = [{**f,'tracks':[t for t in f['tracks'] if (t['id'],f['frame_index']) not in skipped]}
                for f in frames]
    return filtered,assignment,applied


def geometry_candidates(tracks, frames, edges):
    raw = {(a, b): appearance for appearance, a, b, _ in edges}
    collisions = defaultdict(list)
    pts = {}
    for frame in frames:
        frame_key = frame['camera'], frame['frame_index']
        pts[frame_key] = frame['pts']
        for a, b in combinations(sorted(frame['tracks'], key=lambda row: row['id']), 2):
            collisions[a['id'], b['id']].append((frame_key, a, b))
    found = []
    for pair, shared in collisions.items():
        a, b = pair
        if len(tracks[b]) < len(tracks[a]):
            a, b = b, a
            shared = [(frame, right, left) for frame, left, right in shared]
        if (not MIN_COMMON_FRAMES <= len(shared) <= MAX_COMMON_FRAMES
                or len(tracks[b]) < MIN_SUPPORT_RATIO * len(tracks[a])
                or pts[tracks[a][-1][:2]] - pts[tracks[a][0][:2]] > MAX_SHORT_DURATION
                or raw.get(pair, 0.) < MIN_PROTOTYPE):
            continue
        metrics = [(iou(np.asarray(left['box']), np.asarray(right['box'])),
                    containment(np.asarray(left['box']), np.asarray(right['box'])))
                   for _, left, right in shared]
        min_iou, contained = map(min, zip(*metrics, strict=True))
        if min_iou >= MIN_IOU and contained >= MIN_CONTAINMENT:
            found.append({'short': a, 'long': b, 'shared': shared, 'iou_min': min_iou,
                          'containment_min': contained, 'prototype_cosine': raw[pair]})
    return found, pts


def same_frame_keys(candidate):
    return [(crop_key(*frame, a['box']), crop_key(*frame, b['box']))
            for frame, a, b in candidate['shared']]


def supported_continuation(short, long, deleted, pts):
    remainder = [row for row in short if row[:2] not in deleted]
    if not remainder:
        return True
    runs = [[remainder[0]]]
    for row in remainder[1:]:
        if pts[row[:2]] - pts[runs[-1][-1][:2]] > .65:
            runs.append([])
        runs[-1].append(row)

    def close(a, b):
        if abs(pts[a[:2]] - pts[b[:2]]) > .8:
            return False
        first, second = np.asarray(a[3]).reshape(2, 2), np.asarray(b[3]).reshape(2, 2)
        height = max(1., (first[1, 1] - first[0, 1] + second[1, 1] - second[0, 1]) / 2)
        return np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)) <= .35 * height

    return all(any(close(edge, other) for edge in (run[0], run[-1]) for other in long) for run in runs)


def select_support(choices, assignment, far, tracks, margin):
    # One strongest actual transition per pair of groups. Pairs remain subject
    # to camera/frame occupancy and to every recorded distant co-observation.
    group_members, occupied=defaultdict(set),defaultdict(set)
    for a,g in assignment.items():
        group_members[g].add(a)
        occupied[g].update(row[:2] for row in tracks[a])
    neighbors=defaultdict(dict)
    for score,row in sorted(choices,key=lambda p:(-p[0],p[1]['a'],p[1]['b'])):
        a,b=row['a'],row['b']
        ga,gb=assignment[a],assignment[b]
        if ga==gb or gb in neighbors.get(ga, ()): continue
        if not occupied[ga].isdisjoint(occupied[gb]): continue
        if any(tuple(sorted((i,j))) in far for i in group_members[ga] for j in group_members[gb]):
            continue
        neighbors[ga][gb]=neighbors[gb][ga]=(score,row)
    options={g: sorted(values.items(),key=lambda p:(-p[1][0],p[0])) for g,values in neighbors.items()}
    use=[]
    for ga,rows in options.items():
        gb,(score,row)=rows[0]
        if ga>=gb or options[gb][0][0]!=ga: continue
        runner_up=max(rows[1][1][0] if len(rows)>1 else 0.,
                      options[gb][1][1][0] if len(options[gb])>1 else 0.)
        if score-runner_up >= margin: use.append((score,row))
    use.sort(key=lambda p:(-p[0],p[1]['a'],p[1]['b']))
    # Since matching is reciprocal, each group participates at most once in
    # this round. A second round would need new evidence/ambiguity checks.
    result,accepted=join(tracks,assignment,[(s,r['a'],r['b'],r) for s,r in use])
    return result,[dict(row,score=score) for score,_,_,row in accepted]


def rectangle(box):
    x1,y1,x2,y2 = box
    return np.asarray([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], dtype=np.float32)


def key(a, b):
    return tuple(sorted((a, b)))


def members_by_group(assignment):
    groups = defaultdict(set)
    for local, group in assignment.items():
        groups[group].add(local)
    return groups


def mirror_coverage(box, polygons):
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max((cv2.intersectConvexConvex(rectangle(box), poly)[0] / max(1, area) for poly in polygons), default=0.)


def reflection_repair(frames, assignment, raw_edges, weighted, forbidden, polygons):
    tracks = collect_tracks(frames)
    groups = members_by_group(assignment)
    occupied = {g: {r[:2]: (local, r[3]) for local in group for r in tracks[local]}
                for g, group in groups.items()}
    raw = {key(a,b): sim for sim,a,b,_ in raw_edges}
    weights = {key(a,b): score for score,a,b,_ in weighted}
    cover = {(local, r[:2]): mirror_coverage(r[3], polygons.get(r[0], []))
             for local, rows in tracks.items() for r in rows if r[0] in polygons}
    options = []
    for a,b in combinations(sorted(groups),2):
        collisions = occupied[a].keys() & occupied[b].keys()
        if len(collisions) < 2 or any(camera not in polygons for camera,_ in collisions):
            continue
        if any(key(l,r) in forbidden for l in groups[a] for r in groups[b]):
            continue
        edges = [weights[key(l,r)] for l in groups[a] for r in groups[b] if key(l,r) in weights]
        if not edges or mean(edges) < .25:
            continue
        drop = set()
        for frame in collisions:
            left = occupied[a][frame][0]
            right = occupied[b][frame][0]
            va, vb = cover[left,frame], cover[right,frame]
            if raw.get(key(left,right), 0.) < .87:
                break
            if va >= .65 and vb <= .20:
                drop.add((left,frame))
            elif vb >= .65 and va <= .20:
                drop.add((right,frame))
            else:
                break
        if len(drop) == len(collisions):
            times = sorted({row[2] for local in groups[a] | groups[b] for row in tracks[local]
                            if (local, row[:2]) not in drop})
            if any(right - left > MAX_GAP for left, right in pairwise(times)):
                continue
            options.append((mean(edges),a,b,drop))
    used, deleted, events = set(),set(),[]
    assignment = dict(assignment)
    for score,a,b,drop in sorted(options, key=lambda item:(-item[0],item[1],item[2])):
        if a in used or b in used:
            continue
        used.update((a,b))
        deleted.update(drop)
        for local in groups[b]:
            assignment[local] = a
        events.append({'groups':[a,b], 'locals':[sorted(groups[a]),sorted(groups[b])],
                       'score':score, 'deleted':sorted(drop)})
    if not deleted:
        return frames, assignment, events
    frames = [{**f,'tracks':[t for t in f['tracks'] if (t['id'],(f['camera'],f['frame_index'])) not in deleted]} for f in frames]
    surviving = collect_tracks(frames)
    return frames, {local:g for local,g in assignment.items() if local in surviving}, events


def rescue(frames, assignment, raw_edges, weighted, topology, forbidden, far):
    tracks = collect_tracks(frames)
    max_length_bins = 25
    groups = members_by_group(assignment)
    bins = {g:{round(r[2]*5) for l in members for r in tracks[l]} for g,members in groups.items()}
    weights = {key(a,b):v for v,a,b,_ in weighted}
    index, adjacent, _ = topology
    options = []
    for app,a,b,gap in raw_edges:
        if a not in tracks or b not in tracks or assignment[a] == assignment[b]:
            continue
        if min(len(bins[assignment[a]]),len(bins[assignment[b]])) > max_length_bins:
            continue
        if app < .88 or weights[key(a,b)] < .42 or not .01 <= gap <= 2.5:
            continue
        earlier,later = (a,b) if tracks[a][0][2]<tracks[b][0][2] else (b,a)
        ca,cb=tracks[earlier][0][0],tracks[later][0][0]
        if ca==cb or not adjacent[index[ca],index[cb]]:
            continue
        value = .55*app + .45*weights[key(a,b)] - .025*gap
        options.append((value,{'a':a,'b':b,'gap':gap,'appearance':app,'edge':weights[key(a,b)]}))
    current = add_conflicts(tracks, {pair for pair in forbidden if set(pair) <= tracks.keys()})
    assignment, events = select_support(options, assignment, far, current, .045)
    return frames, assignment, events


def crop_key(camera, frame, box):
    return camera, frame, tuple(np.rint(box).astype(int).tolist())


def retain_tracks(frames, predicate):
    return [{**frame, 'tracks': [track for track in frame['tracks'] if predicate(frame, track)]}
            for frame in frames]


def early_dedup(tracks, frames, options, known, pts, forbidden):
    seeds = {local: local for local in tracks}
    consumed, dropped = set(), set()
    for option in sorted(options, key=lambda row: (-row['prototype_cosine'], row['short'])):
        a, b = option['short'], option['long']
        if consumed & {a, b} or key(a, b) in forbidden:
            continue
        shared = option['shared']
        cosines = [float(np.dot(known[left], known[right])) for left, right in same_frame_keys(option)]
        if sum(value >= MIN_PAIR_SIMILARITY for value in cosines) / len(cosines) < MIN_FRACTION_NEAR:
            continue
        short_deleted = {frame for frame, short, long in shared if short['score'] <= long['score']}
        if len(short_deleted) / len(shared) < .75:
            continue
        long_deleted = {frame for frame, _, _ in shared} - short_deleted
        if not supported_continuation(tracks[a], [r for r in tracks[b] if r[:2] not in long_deleted],
                                      short_deleted, pts):
            continue
        times = sorted({r[2] for r in tracks[a] if r[:2] not in short_deleted}
                       | {r[2] for r in tracks[b] if r[:2] not in long_deleted})
        if any(right - left > MAX_GAP for left, right in pairwise(times)):
            continue
        consumed.update((a, b))
        dropped.update((a, *frame) for frame in short_deleted)
        dropped.update((b, *frame) for frame in long_deleted)
        seeds[a] = b
    if not dropped:
        return frames, tracks, seeds
    frames = retain_tracks(frames, lambda f, t: (t['id'], f['camera'], f['frame_index']) not in dropped)
    tracks = collect_tracks(frames)
    return frames, tracks, {local: seeds[local] for local in tracks}


def remove_shape_fragments(frames, tracks, seeds):
    removed = set()
    for local, rows in tracks.items():
        if len(rows) < 2 or rows[-1][2] - rows[0][2] > 2.:
            continue
        aspects = np.array([(box[2] - box[0]) / max(1., box[3] - box[1]) for _, _, _, box in rows])
        if aspects.mean() > .60 or aspects.std() > .15:
            removed.add(local)
    if not removed:
        return frames, tracks, seeds
    frames = retain_tracks(frames, lambda f, t: t['id'] not in removed)
    tracks = {local: rows for local, rows in tracks.items() if local not in removed}
    seed_members = defaultdict(list)
    for local in sorted(tracks):
        seed_members[seeds[local]].append(local)
    seeds = {local: root if root in tracks else members[0]
             for root, members in seed_members.items() for local in members}
    return frames, tracks, seeds


def late_overlap(simultaneous, far, tracks, assignment):
    choices = [(.55 * row['app'] + .45 * max(0., 1 - row['median'] / 36), row)
               for row in simultaneous if row['app'] >= .83 and row['median'] <= 12.
               and row['p80'] <= 18. and row['near'] >= .7]
    return select_support(choices, assignment, far, tracks, .04)
