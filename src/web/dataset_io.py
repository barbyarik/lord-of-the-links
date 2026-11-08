"""Read the team contract; keep canonical records separate from UI projections."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from math import isfinite
from pathlib import Path

class DatasetError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise DatasetError(message)

def bundle_file(root, path, suffixes=None):
    """Resolve bundle references without following links outside the selected root."""
    path = Path(path)
    if path.is_absolute():
        return None
    target = (root / path).resolve()
    if not target.is_relative_to(root) or (suffixes is not None and target.suffix.lower() not in suffixes):
        return None
    return target if target.is_file() else None


def read_csv(path, required=()):
    require(path.is_file(), f'Missing file: {path}')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and set(required) <= set(reader.fieldnames),
                f'{path.name}: required columns missing: {sorted(set(required)-set(reader.fieldnames or []))}')
        require(len(reader.fieldnames) == len(set(reader.fieldnames)), f'{path.name}: duplicate column names')
        rows = list(reader)
        require(all(None not in row and all(value is not None for value in row.values()) for row in rows),
                f'{path.name}: row width does not match the header')
        return rows

def integer(value, label, nullable=False):
    if value in ('', None) and nullable:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise DatasetError(f'{label}: expected integer, got {value!r}') from None

def time_value(value):
    if not value:
        return None
    require(isinstance(value, str) and 'T' in value, f'Invalid clock_time: {value!r}')
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise DatasetError(f'Invalid clock_time: {value!r}') from None
    require(dt.tzinfo is None, 'clock_time must not be assigned a timezone')
    return dt

def typed(rows, integers, nullable=()):
    out = []
    for row in rows:
        r = {k: (None if v == '' else v) for k, v in row.items()}
        for k in integers:
            r[k] = integer(r.get(k), k, k in nullable)
        out.append(r)
    return out

FRAME_FIELDS = 'camera_id sample_ms frame_index pts_ms clock_time clock_status image review_status person_count'.split()
OBS_FIELDS = 'camera_id sample_ms frame_index pts_ms clock_time clock_status local_id global_id x_min y_min x_max y_max review_status'.split()
CAM_FIELDS = 'camera_id plan_label source sha256 container_duration_s first_decoded_pts_ms reported_fps width height map_evidence_frame map_evidence'.split()
CLOCKS = {'ocr', 'manual', 'estimated', 'estimated_from_1fps', 'unreadable', 'missing'}


def validate_frames(frames, observations, cameras, strict=True, image_root=None, bundle_root=None):
    by_key, obs_keys, counts = {}, set(), Counter()
    for f in frames:
        key = f['camera_id'], f['sample_ms']
        require(key not in by_key, f'Duplicate frame key {key}')
        require(f['camera_id'] in cameras, f'Unknown camera {key}')
        require(0 <= f['sample_ms'] <= 1199500 and f['sample_ms'] % 500 == 0, f'Invalid sample_ms {key}')
        require(f['review_status'] in {'reviewed', 'unreviewed', 'missing'}, f'Unknown review_status at {key}')
        require(f['clock_status'] in CLOCKS, f'Unknown clock_status at {key}')
        time_value(f.get('clock_time'))
        if f['review_status'] == 'missing':
            require(all(f.get(k) is None for k in ('pts_ms', 'frame_index', 'image', 'person_count', 'clock_time')), f'Missing frame has data at {key}')
            require(f['clock_status'] == 'missing', f'Missing frame has wrong clock_status at {key}')
        else:
            require(f['clock_status'] != 'missing', f'Available frame has missing clock_status at {key}')
            require(f['pts_ms'] is not None and f['frame_index'] is not None, f'Available frame lacks PTS/index at {key}')
            require(0 <= f['pts_ms']-f['sample_ms'] <= 120, f'PTS outside [sample_ms, sample_ms+120] at {key}')
            require(f['frame_index'] >= 0, f'Negative frame_index at {key}')
            if image_root is not None:
                relative = image_root.relative_to(bundle_root) / f['image'] if f.get('image') else None
                require(relative is not None and bundle_file(bundle_root, relative, {'.jpg', '.jpeg', '.png'}),
                        f'Invalid or missing preview at {key}')
            if f['review_status'] == 'reviewed':
                require(f['person_count'] is not None and f['person_count'] >= 0, f'Reviewed frame lacks person_count at {key}')
            else:
                require(f['person_count'] is None, f'Unreviewed frame has person_count at {key}')
        if f['clock_status'] in {'missing', 'unreadable'}:
            require(f.get('clock_time') is None, f'Unknown clock has timestamp at {key}')
        else:
            require(bool(f.get('clock_time')), f'Clock timestamp missing at {key}')
        by_key[key] = f
    if strict:
        expected = {(c, t) for c in range(1, 6) for t in range(0, 1200000, 500)}
        require(set(by_key) == expected, 'Contract requires the complete 5 x 2400 grid, including missing frames')
    for o in observations:
        key = o['camera_id'], o['sample_ms']
        unique = (*key, o['local_id'])
        require(bool(o['local_id']) and unique not in obs_keys, f'Duplicate/empty observation key {unique}')
        obs_keys.add(unique)
        require(key in by_key, f'Observation without frame {unique}')
        f = by_key[key]
        require(f['review_status'] == o['review_status'] == 'reviewed', f'Observation on a non-reviewed frame {unique}')
        for k in ('pts_ms', 'frame_index', 'clock_time', 'clock_status'):
            require(o.get(k) == f.get(k), f'Observation/frame disagree on {k}: {unique}')
        cam = cameras[o['camera_id']]
        require(0 <= o['x_min'] < o['x_max'] <= cam['width'] and 0 <= o['y_min'] < o['y_max'] <= cam['height'], f'Invalid bbox {unique}')
        counts[key] += 1
    for key, f in by_key.items():
        if f['review_status'] == 'reviewed':
            require(counts[key] == f['person_count'], f'person_count does not match observations at {key}')
    return by_key


def validate_tracks(export, observations, frames, strict=True):
    grouped = defaultdict(list)
    local_identities = {}
    for o in observations:
        if strict:
            local = o['local_id']
            identity = (o['camera_id'], o.get('global_id'))
            require(local not in local_identities or local_identities[local] == identity,
                    f'Local episode changes camera/global_id: {local}')
            local_identities[local] = identity
        grouped[(o.get('global_id') or o['local_id']) if strict else o['track_id']].append(o)
    seen, covered = set(), set()
    total = 0
    for tr in export['tracks']:
        ident = tr.get('id')
        require(bool(ident) and ident not in seen, f'Duplicate/empty track id {ident}')
        seen.add(ident)
        require(ident in grouped, f'Track without observations: {ident}')
        if strict:
            linked = any(o.get('global_id') for o in grouped[ident])
            require(tr.get('link_status') == ('linked_visual' if linked else 'local_only'), f'Wrong link_status for {ident}')
        for seg in tr['segments']:
            c, local = seg['camera_id'], seg['local_id']
            lo, hi = seg['first_sample_ms'], seg['last_sample_ms']
            require(type(lo) is int and type(hi) is int and lo <= hi, f'Invalid segment bounds for {ident}')
            rows = sorted((o for o in grouped[ident] if o['camera_id'] == c and o['local_id'] == local and lo <= o['sample_ms'] <= hi), key=lambda o:o['sample_ms'])
            require(bool(rows) and rows[0]['sample_ms'] == lo and rows[-1]['sample_ms'] == hi, f'Segment endpoints do not match observations for {ident}/{local}')
            require(all(b['sample_ms']-a['sample_ms'] <= 500 for a,b in zip(rows,rows[1:])), f'Segment crosses missing observations: {ident}/{local}')
            require(type(seg['n_reviewed_observations']) is int and seg['n_reviewed_observations'] == len(rows), f'Segment count mismatch for {ident}/{local}')
            for name, row in [('first',rows[0]),('last',rows[-1])]:
                require(seg[name+'_visible_pts_ms'] == row['pts_ms'], f'Segment PTS mismatch for {ident}/{local}')
                require((seg.get(name+'_clock_time') or None) == row.get('clock_time'), f'Segment clock mismatch for {ident}/{local}')
            for row in rows:
                key = row['camera_id'], row['sample_ms'], row['local_id']
                require(key not in covered, f'Overlapping source segments duplicate {key}')
                covered.add(key)
            total += len(rows)
    require(seen == set(grouped), 'tracks.json does not cover all source IDs')
    require(total == len(observations), 'Sum of segment observations does not match observations.csv')
    require(type(export['sampling_hz']) in (int, float) and export['sampling_hz'] == 2 and type(export['sample_count']) is int and export['sample_count'] == len(frames), 'Wrong sampling_hz/sample_count')
    actual = Counter(f['review_status'] for f in frames.values())
    require(all(export['frame_status_counts'].get(k,0) == actual.get(k,0) for k in set(actual)|set(export['frame_status_counts'])), 'frame_status_counts mismatch')


def read_export(path):
    export = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(export, dict), 'tracks.json must contain an object')
    require(isinstance(export.get('tracks'), list), 'tracks.json: tracks must be an array')
    counts = export.get('frame_status_counts')
    require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()),
            'tracks.json: invalid frame_status_counts')
    segment_fields = {
        'camera_id', 'local_id', 'first_sample_ms', 'last_sample_ms',
        'first_visible_pts_ms', 'last_visible_pts_ms', 'first_clock_time',
        'last_clock_time', 'n_reviewed_observations',
    }
    for track in export['tracks']:
        require(isinstance(track, dict) and isinstance(track.get('id'), str) and bool(track['id']),
                'tracks.json: invalid track object/id')
        require(isinstance(track.get('segments'), list), 'tracks.json: segments must be an array')
        for segment in track['segments']:
            require(isinstance(segment, dict) and segment_fields <= segment.keys(),
                    f'tracks.json: segment fields missing for {track["id"]}')
            require(type(segment['camera_id']) is int and isinstance(segment['local_id'], str),
                    f'tracks.json: invalid segment camera/local ID for {track["id"]}')
    return export


def load_dataset(root):
    root = Path(root).resolve()
    # The manifest declares the MP4 handoff extension; raw contract files never need it.
    legacy = (root/'manifest.json').is_file()
    annotation_root = root if legacy or (root/'frames.csv').is_file() else root/'dataset-2fps'
    require((annotation_root/'frames.csv').is_file(), 'Expected dataset-2fps/frames.csv, or a flat contract export')
    strict = not legacy
    frames = typed(read_csv(annotation_root/'frames.csv', FRAME_FIELDS), ('camera_id','sample_ms','pts_ms','frame_index','person_count'), ('pts_ms','frame_index','person_count'))
    observations = typed(read_csv(annotation_root/'observations.csv', OBS_FIELDS), ('camera_id','sample_ms','pts_ms','frame_index','x_min','y_min','x_max','y_max'))
    export = read_export(annotation_root / 'tracks.json')
    if legacy:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
        cam_rows = read_csv(root/'cameras.csv')
        cameras = {int(c['camera_id']): {'camera_id':int(c['camera_id']), 'width':int(c['bbox_width']), 'height':int(c['bbox_height']), **c} for c in manifest['cameras']}
        for c in cameras.values(): c['width']=int(c['bbox_width']);c['height']=int(c['bbox_height'])
        reviews = {}; warnings = ['MP4 handoff extension: episode IDs and media offsets are supplied by its manifest.']
        camera_root = root
    else:
        camera_path = (annotation_root.parent/'dataset'/'cameras.csv')
        if not camera_path.exists(): camera_path=annotation_root/'cameras.csv'
        cam_rows = typed(read_csv(camera_path,CAM_FIELDS), ('camera_id','width','height','first_decoded_pts_ms'))
        cameras = {}
        for c in cam_rows:
            require(c['camera_id'] not in cameras, 'Duplicate camera_id')
            require(c['width'] > 0 and c['height'] > 0, 'Invalid camera dimensions')
            try:
                require(Fraction(c['reported_fps']) > 0, 'Invalid reported_fps')
                c['container_duration_s'] = float(c['container_duration_s'])
                require(isfinite(c['container_duration_s']) and c['container_duration_s'] > 0, 'Invalid container_duration_s')
            except (ValueError, ZeroDivisionError): raise DatasetError('Invalid FPS or duration') from None
            require(len(c['sha256'] or '')==64 and all(x in '0123456789abcdefABCDEF' for x in c['sha256']), 'Invalid source SHA-256')
            cameras[c['camera_id']] = c
        require(set(cameras)==set(range(1,6)), 'Contract requires cameras 1..5')
        require(all(k in export for k in ('annotation_method','clock_alignment','identity_coverage','interval_semantics','tracks')), 'tracks.json lacks contract metadata')
        require(export.get('release_version')=='2026-11-07-v1', 'Unsupported raw contract release_version')
        reviews = {'identity_review':read_csv(annotation_root/'identity_review.csv', ['local_id']), 'links_review':read_csv(annotation_root/'links_review.csv',['left_id','right_id'])}
        ids=[r['local_id'] for r in reviews['identity_review']]
        require(all(ids) and len(ids)==len(set(ids)), 'Duplicate/empty identity_review.local_id')
        links = [(r['left_id'], r['right_id']) for r in reviews['links_review']]
        require(all(left for left, _ in links) and len(links) == len(set(links)),
                'Duplicate/empty links_review key')
        require(set(o['local_id'] for o in observations)<=set(ids), 'identity_review does not cover local episodes')
        warnings = ['clock_time has no timezone and does not guarantee cross-camera synchronization.']
        camera_root=camera_path.parent
        manifest = None
    index = validate_frames(frames, observations, cameras, strict, annotation_root if strict else None, root)
    validate_tracks(export, observations, index, strict)
    return {'root':root,'kind':'contract' if strict else 'handoff','annotation_root':annotation_root,'camera_root':camera_root,'frames':frames,'observations':observations,'export':export,'cameras':cameras,'camera_rows':cam_rows,'reviews':reviews,'manifest':manifest,'warnings':warnings}
