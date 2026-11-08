"""Association primitives adapted from the measured MTMC pipeline."""
import heapq
from collections import defaultdict

import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.special import expit

from .geometry import edge_positions

CROPS_PER_TRACK = 8
MAX_GAP = 60.
THRESHOLD = .25


def visible_interval(rows):
    times = [time for camera, _, time, _ in rows if camera > 0]
    return min(times), max(times)


def interval_gap(left, right):
    return max(0., left[0] - right[1], right[0] - left[1])


def union_interval(left, right):
    return min(left[0], right[0]), max(left[1], right[1])


def collect_tracks(frames):
    tracks = defaultdict(list)
    for frame in frames:
        for track in frame['tracks']:
            tracks[track['id']].append((frame['camera'], frame['frame_index'], frame['time'], track['box']))
    return dict(tracks)


def choose_crops(observations):
    # Reject unusably small crops only when a better view exists for this track.
    usable = [row for row in observations if row[3][2] - row[3][0] >= 12 and row[3][3] - row[3][1] >= 24]
    options = usable or observations
    if len(options) <= CROPS_PER_TRACK:
        return options
    times = np.array([row[2] for row in options])
    available = np.ones(len(times), dtype=bool)
    selected = []
    for target in np.linspace(times[0], times[-1], CROPS_PER_TRACK):
        index = int(np.argmin(np.where(available, np.abs(times - target), np.inf)))
        available[index] = False
        selected.append(options[index])
    return selected


def candidate_edges(tracks, vectors):
    prototypes = {}
    for track_id, entries in vectors.items():
        prototype = np.mean([entry['embedding'] for entry in entries], axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototypes[track_id] = prototype / norm
    names = sorted(prototypes)
    if not names:
        return []
    matrix = np.stack([prototypes[name] for name in names])
    similarities = matrix @ matrix.T
    ends = {name: (tracks[name][0][2], tracks[name][-1][2]) for name in names}
    edges = []
    for i, left in enumerate(names):
        a0, a1 = ends[left]
        for j in range(i + 1, len(names)):
            right = names[j]
            b0, b1 = ends[right]
            gap = max(0.0, b0 - a1, a0 - b1)
            if gap <= MAX_GAP:
                edges.append((float(np.clip(similarities[i, j], -1, 1)), left, right, gap))
    return sorted(edges, key=lambda edge: (-edge[0], edge[1], edge[2]))


def agglomerate(tracks, edges, threshold, *, gap_weight=None, edge_importance=None, merge_allowed=None):
    """Greedily merge the highest scoring compatible groups using edge averages.

    The optional weighting function returns positive importance for a gap or
    for an edge, including its endpoints.
    """
    if gap_weight is not None and edge_importance is not None:
        raise ValueError('Specify only one edge weighting function')
    members = {identity: {identity} for identity in tracks}
    occupancy = {identity: {(camera, frame) for camera, frame, *_ in rows}
                 for identity, rows in tracks.items()}
    intervals = {identity: visible_interval(rows) for identity, rows in tracks.items()}
    neighbors = {identity: {} for identity in tracks}
    queue = []

    def add(left, right, total, weight_sum):
        left, right = sorted((left, right))
        stats = total, weight_sum
        neighbors[left][right] = neighbors[right][left] = stats
        average = total / weight_sum
        if average >= threshold:
            heapq.heappush(queue, (-average, left, right, total, weight_sum))

    for score, left, right, gap in edges:
        if edge_importance is not None:
            importance = edge_importance(left, right, gap)
        else:
            importance = 1 if gap_weight is None else gap_weight(gap)
        add(left, right, score * importance, importance)

    while queue:
        _, left, right, total, weight_sum = heapq.heappop(queue)
        if left not in members or right not in members or neighbors[left].get(right) != (total, weight_sum):
            continue
        if not occupancy[left].isdisjoint(occupancy[right]):
            continue
        if interval_gap(intervals[left], intervals[right]) > MAX_GAP:
            continue
        if merge_allowed is not None and not merge_allowed(members[left], members[right], weight_sum):
            continue
        right_edges = neighbors.pop(right)
        neighbors[left].pop(right, None)
        for other, (score_sum, n) in right_edges.items():
            if other == left:
                continue
            neighbors[other].pop(right, None)
            previous, m = neighbors[left].get(other, (0., 0))
            add(left, other, previous + score_sum, m + n)
        members[left].update(members.pop(right))
        occupancy[left].update(occupancy.pop(right))
        intervals[left] = union_interval(intervals[left], intervals.pop(right))
    return {identity: group for group, identities in members.items() for identity in identities}


def join(tracks,assignment,proposals):
    members,occupancy = defaultdict(set),defaultdict(set)
    root = dict(assignment)
    for local,group in root.items():
        members[group].add(local)
        occupancy[group].update((camera,frame) for camera,frame,*_ in tracks[local])
    intervals = {group: visible_interval([row for local in locals_ for row in tracks[local]])
                 for group, locals_ in members.items()}
    accepted = []
    for row in proposals:
        _,a,b,*_ = row
        left,right = sorted((root[a],root[b]))
        if left==right or not occupancy[left].isdisjoint(occupancy[right]):
            continue
        if interval_gap(intervals[left], intervals[right]) > MAX_GAP:
            continue
        for local in members[right]:
            root[local] = left
        members[left].update(members.pop(right))
        occupancy[left].update(occupancy.pop(right))
        intervals[left] = union_interval(intervals[left], intervals.pop(right))
        accepted.append(row)
    return root,accepted


def prestitch(tracks,edges,seeds,*,threshold=THRESHOLD):
    if all(local==group for local,group in seeds.items()):
        return agglomerate(tracks,edges,threshold)
    grouped = defaultdict(list)
    for local,rows in tracks.items():
        grouped[seeds[local]].extend(rows)
    weights = defaultdict(lambda:[0.,0])
    for score,a,b,_ in edges:
        left,right = sorted((seeds[a],seeds[b]))
        if left==right:
            continue
        weights[left,right][0] += score
        weights[left,right][1] += 1
    quotient = [(total/n,a,b,0) for (a,b),(total,n) in weights.items()]
    coarse = agglomerate(grouped,quotient,threshold,edge_importance=lambda a,b,_:weights[a,b][1])
    return {local:coarse[group] for local,group in seeds.items()}


def add_conflicts(tracks, pairs):
    tracks={t:list(rows) for t,rows in tracks.items()}
    for i,(a,b) in enumerate(sorted(pairs)):
        phantom=(-2,i,0.,None)
        tracks[a].append(phantom)
        tracks[b].append(phantom)
    return tracks


def endpoint_neighbors(tracks, frames):
    """Find other local boxes near each track's start and end."""
    by_frame={(row['camera'],row['frame_index']):{t['id']:np.asarray(t['box']) for t in row['tracks']}
              for row in frames}

    def neighbors_near(identity,at_start):
        rows=tracks[identity]
        edge=rows[0 if at_start else -1][2]
        possible=[]
        ordered=rows if at_start else rows[::-1]
        for c,index,t,box in ordered[:5]:
            if abs(t-edge)>.85:
                continue
            box=np.asarray(box)
            center=(box[:2]+box[2:])/2
            height=box[3]-box[1]
            for other,ob in by_frame[c,index].items():
                if identity==other:
                    continue
                other_h=ob[3]-ob[1]
                normal=np.sqrt(height*other_h)
                if normal>0 and np.linalg.norm(center-(ob[:2]+ob[2:])/2)/normal<.85:
                    possible.append(other)
        return set(possible)
    return {identity:(neighbors_near(identity,True),neighbors_near(identity,False))
            for identity in tracks}


def replace_weights(edges, x, model):
    logits = ((x - model['mean']) / model['scale']) @ model['weights'] + model['intercept']
    return [(float(p), a, b, gap) for p, (_, a, b, gap) in zip(expit(logits), edges, strict=True)]



def camera_topology(cameras, edges):
    index = {camera: i for i, camera in enumerate(sorted(cameras))}
    adjacent = np.zeros((len(index), len(index)))
    for left, right in edges:
        if left in index and right in index:
            adjacent[index[left], index[right]] = 1
    return index, adjacent, shortest_path(adjacent, directed=True, unweighted=True)


def edge_features(tracks, edges, geometry, topology, image_sizes):
    index, adjacent, distance = topology
    position = edge_positions(tracks, edges, geometry, image_sizes)
    heights = {local: np.median([box[3] - box[1] for _, _, _, box in rows])
               for local, rows in tracks.items()}
    features = []
    for appearance, left, right, gap in edges:
        a, b = tracks[left], tracks[right]
        ca, cb = index[a[0][0]], index[b[0][0]]
        if b[-1][2] < a[0][2]:
            ca, cb = cb, ca
        overlap = max(0., min(a[-1][2], b[-1][2]) - max(a[0][2], b[0][2]))
        score, _, samples = position.get((left, right), (0., 0., 0))
        hops = min(len(index), distance[ca, cb])
        features.append([appearance, np.log1p(gap), np.log1p(overlap), ca == cb,
                         adjacent[ca, cb], hops, bool(samples), score, np.log1p(samples),
                         np.log1p(min(heights[left], heights[right]))])
    return np.asarray(features, dtype=float).reshape(-1, 10)


def score_edges(tracks, frames, edges, calibration, topology, image_sizes, *, penalty):
    features = edge_features(tracks, edges, calibration.get('pair_geometry', {}), topology, image_sizes)
    if calibration.get('cosine_quantiles'):
        features[:, 0] = np.interp(features[:, 0], *calibration['cosine_quantiles'])
    weighted = replace_weights(edges, features, calibration['model'])
    if not penalty:
        return weighted
    endpoints = endpoint_neighbors(tracks, frames)
    adjusted = []
    for weight, a, b, gap in weighted:
        first, last = sorted((a, b), key=lambda t: (tracks[t][0][2], t))
        crowded = (tracks[a][0][0] == tracks[b][0][0] and
                   (endpoints[first][1] or endpoints[last][0]))
        adjusted.append((max(0., weight - .15) if crowded else weight, a, b, gap))
    return adjusted
