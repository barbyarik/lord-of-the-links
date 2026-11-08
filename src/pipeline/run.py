"""Run reusable MTMC inference from synchronized videos or portable caches."""
import json
from collections import defaultdict
from collections.abc import Mapping
from hashlib import file_digest
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile

import numpy as np

from .geometry import geometry_evidence, strict_conflicts
from .graph import (
    MAX_GAP, add_conflicts, camera_topology, candidate_edges, choose_crops,
    collect_tracks, join, prestitch, score_edges,
)
from .profiles import load_profile
from .repairs import (
    candidates, crop_key, deduplicate, early_dedup, geometry_candidates, late_overlap,
    reciprocal, rectangle, reflection_repair, remove_shape_fragments, rescue,
    retain_tracks, same_frame_keys,
)
from .scene import covered_groups, discover_scene, grouped_boxes, mirror_vote_groups, recurring_static
from .vision import Appearance, SAMPLING_HZ, detect_camera, load_yolo, track_camera


def read_object(source):
    return dict(source) if isinstance(source, Mapping) else json.loads(Path(source).read_text())


def read_detection_bundle(source):
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        source = {int(path.stem.removeprefix('camera-')): read_object(path)
                  for path in Path(source).glob('camera-*.json')}
    else:
        source = read_object(source)
    scene = source.get('scene') if 'cameras' in source else None
    cameras = source.get('cameras', source)
    return {int(camera): read_object(data) for camera, data in cameras.items()}, scene


def read_appearance_bundle(source):
    if source is None:
        return {}, {}
    if isinstance(source, Mapping):
        return dict(source), {}
    with np.load(source, allow_pickle=False) as cache:
        keys, vectors = cache['keys'], cache['vectors']
        if keys.ndim != 2 or keys.shape[1] != 6 or not np.issubdtype(keys.dtype, np.integer):
            raise ValueError('Appearance cache keys must be an integer array with shape (N, 6)')
        if vectors.shape != (len(keys), 1280) or not np.isfinite(vectors).all():
            raise ValueError('Appearance cache vectors must be finite with shape (N, 1280)')
        by_crop = {(camera, frame, (x1, y1, x2, y2)): vector
                   for (camera, frame, x1, y1, x2, y2), vector in zip(keys.tolist(), vectors, strict=True)}
        if len(by_crop) != len(keys):
            raise ValueError('Appearance cache has duplicate crop keys')
        sources = {}
        if 'source_cameras' in cache:
            sources = dict(zip(cache['source_cameras'].tolist(), cache['source_sha256'].tolist(), strict=True))
        return by_crop, sources


def video_fingerprint(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def scene_requirements(enabled):
    need_coco = bool(enabled & {'screens', 'semantic_zones'})
    need_mirrors = bool(enabled & {'reflection', 'semantic_zones', 'mirror_votes'})
    return {name for name, needed in [('coco', need_coco), ('mirrors', need_mirrors)] if needed}


def resolve_scene(site, cached_scene, enabled, cached, videos, yolo, model_dir, device, progress):
    required = scene_requirements(enabled)
    if not required:
        return {}, {}
    scene = site.get('scene')
    if scene is None and cached_scene and required <= set(cached_scene.get('capabilities', required)):
        scene = cached_scene
    if scene is not None:
        return ({int(camera): rows for camera, rows in scene.get('regions', {}).items()},
                {int(camera): rows for camera, rows in scene.get('mirror_votes', {}).items()})
    if 'coco' in required and yolo is None and set(cached) <= videos.keys():
        yolo = load_yolo(model_dir)
    return discover_scene(cached, videos, yolo, model_dir, device, progress,
                          need_coco='coco' in required, need_mirrors='mirrors' in required)

def keep_groups(frames, tracks, assignment, rules, stable, votes, start):
    groups = grouped_boxes(tracks, assignment)
    removed = set()
    if 'semantic_zones' in rules:
        removed |= covered_groups(groups, stable, {'screen', 'mirror', 'seat', 'table'})
    if 'static' in rules:
        removed |= recurring_static(groups)
    if 'mirror_votes' in rules:
        removed |= mirror_vote_groups(groups, votes)
    if 'short_group' in rules:
        scores = defaultdict(dict)
        for frame in frames:
            tick = round((frame['time'] - start) * SAMPLING_HZ)
            for track in frame['tracks']:
                group = assignment[track['id']]
                scores[group][tick] = max(scores[group].get(tick, 0.), track['score'])
        removed |= {group for group, ticks in scores.items()
                    if len(ticks) < 8 and (len(ticks) < 2 or median(ticks.values()) < .7)}
    return {local: group for local, group in assignment.items() if group not in removed}


def stitch_frames(frames, image_sizes, site, rules, appearance, stable, votes, start):
    tracks = collect_tracks(frames)
    if not tracks:
        return frames, {}
    calibration = site['association']
    # Keep the installed topology when replaying a camera subset.
    cameras = set(image_sizes) | {int(camera) for camera in site.get('cameras', {})}
    adjacency = site.get('adjacency', ())
    cameras.update(camera for pair in adjacency for camera in pair)
    topology = camera_topology(cameras, adjacency)
    gallery = {local: [crop_key(camera, index, box) for camera, index, _, box in choose_crops(rows)]
               for local, rows in tracks.items()}
    appearance.ensure({key for crops in gallery.values() for key in crops})
    vectors = {local: [{'embedding': appearance.known[key]} for key in crops]
               for local, crops in gallery.items()}
    raw = candidate_edges(tracks, vectors)
    weighted = score_edges(tracks, frames, raw, calibration, topology, image_sizes,
                           penalty='penalty' in rules)

    geometry = site.get('geometry', {})
    guards = site.get('guards')
    needs_geometry = bool(rules & {'strict_floor', 'late_overlap', 'far_veto'})
    simultaneous, far = (geometry_evidence(tracks, frames, raw, geometry, image_sizes, guards)
                         if needs_geometry else ([], set()))
    forbidden = strict_conflicts(simultaneous, far) if 'strict_floor' in rules else set()
    seeds = {local: local for local in tracks}
    if 'early_dedup' in rules:
        options, pts = geometry_candidates(tracks, frames, raw)
        appearance.ensure({key for option in options for pair in same_frame_keys(option) for key in pair})
        proposed_frames, proposed_tracks, proposed_seeds = early_dedup(
            tracks, frames, options, appearance.known, pts, forbidden)
        proposed_simultaneous, proposed_far, proposed_forbidden = simultaneous, far, forbidden
        if proposed_frames is not frames and needs_geometry:
            surviving = [edge for edge in raw if edge[1] in proposed_tracks and edge[2] in proposed_tracks]
            proposed_simultaneous, proposed_far = geometry_evidence(
                proposed_tracks, proposed_frames, surviving, geometry, image_sizes, guards)
            proposed_forbidden = (strict_conflicts(proposed_simultaneous, proposed_far)
                                  if 'strict_floor' in rules else set())
        if all(proposed_seeds[a] != proposed_seeds[b] for a, b in proposed_forbidden):
            frames, tracks, seeds = proposed_frames, proposed_tracks, proposed_seeds
            simultaneous, far, forbidden = proposed_simultaneous, proposed_far, proposed_forbidden
    if 'shape' in rules:
        frames, tracks, seeds = remove_shape_fragments(frames, tracks, seeds)

    raw = [edge for edge in raw if edge[1] in tracks and edge[2] in tracks]
    weighted = [edge for edge in weighted if edge[1] in tracks and edge[2] in tracks]
    forbidden = {pair for pair in forbidden if set(pair) <= tracks.keys()}
    far = {pair for pair in far if set(pair) <= tracks.keys()} if 'far_veto' in rules else set()
    simultaneous = [row for row in simultaneous if row['a'] in tracks and row['b'] in tracks]

    constrained = add_conflicts(tracks, forbidden)
    assignment = prestitch(constrained, weighted, seeds)
    if 'bridge' in rules:
        params = calibration.get('bridge', (.6, .25, .85, .05))
        choices = reciprocal(candidates(tracks, raw, frames), params)
        assignment, _ = join(constrained, assignment, choices)
    if 'boundary_dedup' in rules:
        reduced, proposal, _ = deduplicate(tracks, weighted, frames, assignment, raw)
        if all(proposal[a] != proposal[b] for a, b in forbidden):
            frames, assignment = reduced, proposal
    if 'late_overlap' in rules:
        current = add_conflicts(collect_tracks(frames), forbidden)
        assignment, _ = late_overlap(simultaneous, far, current, assignment)
    if 'reflection' in rules:
        polygons = {camera: [rectangle(region['box']) for region in regions if region['kind'] == 'mirror']
                    for camera, regions in stable.items()}
        polygons = {camera: shapes for camera, shapes in polygons.items() if shapes}
        frames, assignment, _ = reflection_repair(frames, assignment, raw, weighted, forbidden | far, polygons)
    if 'handoff' in rules:
        frames, assignment, _ = rescue(frames, assignment, raw, weighted, topology, forbidden, far)

    assignment = keep_groups(frames, collect_tracks(frames), assignment, rules, stable, votes, start)
    return retain_tracks(frames, lambda frame, track: track['id'] in assignment), assignment


def export_predictions(frames, assignment, start, profile, image_sizes):
    # Enforce the observation gap even if a later repair deleted the last bridging box.
    groups = defaultdict(list)
    for frame in frames:
        for track in frame['tracks']:
            groups[assignment[track['id']]].append((frame['time'], frame['camera'], frame['frame_index'], track))
    segments = []
    for group, rows in groups.items():
        previous = None
        for row in sorted(rows, key=lambda r: (*r[:3], r[3]['id'])):
            if previous is None or row[0] - previous > MAX_GAP:
                segments.append((row[0], group, []))
            segments[-1][2].append(row)
            previous = row[0]
    identifiers = {}
    for index, (_, _, rows) in enumerate(sorted(segments, key=lambda segment: segment[:2]), 1):
        occupied = set()
        for _, camera, frame_index, track in rows:
            if (camera, frame_index) in occupied:
                raise RuntimeError('A global ID has two observations in one camera/source frame')
            occupied.add((camera, frame_index))
            identifiers[camera, frame_index, track['id']] = f'P{index:03d}'

    exported = []
    for frame in frames:
        camera, index = frame['camera'], frame['frame_index']
        tracks = [{**track, 'global_id': identifiers[camera, index, track['id']]} for track in frame['tracks']]
        exported.append({'camera': camera, 'frame_index': index, 'pts': frame['pts'],
                         'time': frame['time'], 'tracks': tracks})
    return {'schema_version': 1, 'profile': profile, 'sampling_hz': SAMPLING_HZ,
            'recording_start': start, 'frames': exported,
            'source_dimensions': {str(camera): [int(width), int(height)]
                                  for camera, (height, width) in sorted(image_sizes.items())}}


def write_json(path, payload):
    path = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            output.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_appearance(path, known, sources):
    path = Path(path)
    crops = sorted(known)
    keys = np.array([(camera, index, *box) for camera, index, box in crops], dtype=np.int64).reshape(-1, 6)
    vectors = np.asarray([known[crop] for crop in crops], dtype=np.float64).reshape(-1, 1280)
    temporary = None
    try:
        with NamedTemporaryFile(mode='wb', dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            cameras = np.asarray(sorted(sources), dtype=np.int64)
            fingerprints = np.asarray([sources[camera] for camera in cameras], dtype='U64')
            np.savez(output, keys=keys, vectors=vectors, source_cameras=cameras, source_sha256=fingerprints)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_pipeline(videos=None, *, site, profile='balanced', detection_cache=None, appearance_cache=None,
                 model_dir=None, output=None, device='cpu', detection_cache_output=None,
                 appearance_cache_output=None, progress=None):
    """Return every 5 Hz source sample with actual boxes and linked identities.

    `videos` maps positive integer camera IDs to source video paths. The optional
    `detection_cache` is a camera mapping or directory of camera-N.json files,
    each containing config.image_size and frames with raw detector rows.
    `appearance_cache` is a trusted mapping or NPZ of exact crop keys and vectors.
    When both caches and scene data are supplied, video and model paths may be omitted.
    Optional cache outputs can be reused directly as inputs; `progress` receives text updates.
    """
    site = read_object(site)
    name, rules = load_profile(profile)
    videos = {int(camera): Path(path) for camera, path in (videos or {}).items()}
    known, appearance_sources = read_appearance_bundle(appearance_cache)
    if progress is None:
        def progress(_message):
            pass

    yolo, cached_scene = None, None
    start = site.get('recording_start')
    end = site.get('recording_end')
    if start is not None and end is not None and not start < end:
        raise ValueError('recording_start must be earlier than recording_end')
    sources = {}
    if detection_cache is None:
        if not videos or any(camera <= 0 for camera in videos):
            raise ValueError('Provide source videos for positive camera IDs or a detection cache')
        yolo = load_yolo(model_dir)
        settings = {int(camera): row for camera, row in site.get('cameras', {}).items()}
        cached = {}
        for camera, path in sorted(videos.items()):
            if detection_cache_output is not None or appearance_cache_output is not None or camera in appearance_sources:
                sources[camera] = video_fingerprint(path)
                if camera in appearance_sources and sources[camera] != appearance_sources[camera]:
                    raise ValueError(f'Camera {camera}: source SHA-256 does not match appearance cache')
            progress(f'Camera {camera}: running YOLO11m')
            data = detect_camera(yolo, path, camera, settings.get(camera, {}), start, end, device, progress)
            if camera in sources:
                data['config']['source_sha256'] = sources[camera]
            cached[camera] = data
    else:
        cached, cached_scene = read_detection_bundle(detection_cache)
        if videos.keys() - cached.keys():
            raise ValueError(f'Detection cache lacks cameras {sorted(videos.keys() - cached.keys())}')
        sources = {camera: data['config']['source_sha256'] for camera, data in cached.items()
                   if data['config'].get('source_sha256')}
        for camera, path in videos.items():
            if camera not in sources:
                raise ValueError(f'Camera {camera}: detection cache has no source SHA-256 to verify video')
            if video_fingerprint(path) != sources[camera]:
                raise ValueError(f'Camera {camera}: source SHA-256 does not match detection cache')
        for camera in sources.keys() & appearance_sources.keys():
            if sources[camera] != appearance_sources[camera]:
                raise ValueError(f'Camera {camera}: source SHA-256 differs between detection and appearance caches')
    if not cached or any(camera <= 0 for camera in cached):
        raise ValueError('At least one positive camera ID is required')
    for camera, data in cached.items():
        if data['config'].get('fps', SAMPLING_HZ) != SAMPLING_HZ:
            raise ValueError(f'Camera {camera}: expected {SAMPLING_HZ} Hz detections')
    if start is None:
        starts = [data['config']['start'] for data in cached.values() if 'start' in data['config']]
        if not starts:
            starts = [frame['time'] for data in cached.values() for frame in data['frames']]
        if not starts:
            raise ValueError('recording_start is required for an empty detection cache')
        start = min(starts)

    if detection_cache is not None:
        bounds = {'start': start}
        if end is not None:
            bounds['end'] = end
        cached = {
            camera: {
                **data,
                'config': data['config'] | bounds,
                'frames': [frame for frame in data['frames']
                           if frame['time'] >= start and (end is None or frame['time'] < end)],
            }
            for camera, data in cached.items()
        }

    stable, votes = resolve_scene(site, cached_scene, rules, cached, videos, yolo, model_dir, device, progress)
    if detection_cache_output is not None:
        scene = {'capabilities': sorted(scene_requirements(rules)), 'regions': stable, 'mirror_votes': votes}
        write_json(detection_cache_output, {'schema_version': 1, 'cameras': cached, 'scene': scene})
    frames = []
    for camera, data in sorted(cached.items()):
        screens = [row['box'] for row in stable.get(camera, ()) if row['kind'] == 'screen'] if 'screens' in rules else []
        progress(f"Camera {camera}: ByteTrack on {len(data['frames'])} sampled frames")
        frames.extend(track_camera(data, camera, screens))
    image_sizes = {camera: data['config']['image_size'] for camera, data in cached.items()}
    pts = {(frame['camera'], frame['frame_index']): frame['pts'] for frame in frames}
    appearance = Appearance(known, videos, pts, model_dir, device, progress)
    progress(f'Stitching {len(frames)} sampled frames using profile {name}')
    frames, assignment = stitch_frames(frames, image_sizes, site, rules, appearance, stable, votes, start)
    if appearance_cache_output is not None:
        write_appearance(appearance_cache_output, appearance.known, sources)
    predictions = export_predictions(frames, assignment, start, name, image_sizes)
    if output is not None:
        write_json(output, predictions)
    return predictions
