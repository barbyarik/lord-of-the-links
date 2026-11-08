"""Build the offline Office UI bundle from canonical pipeline predictions."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

from ..floor_geometry import normalize_site
from .floor import floor_tracks
from .media import media_metadata, prepare_media
from .timing import disambiguate_clock


def build_bundle(predictions: Path, output: Path, *, calibration: Path, media,
                 plan: Path | None = None, media_offsets=None, smoothing_sigma: float = .32) -> Path:
    """Package 5 Hz predictions and native camera media for the local web viewer.

    Offsets are the native first video PTS in seconds. They are detected for MKV
    and default to zero for already browser-playable files.
    """
    prediction = json.loads(Path(predictions).read_text(encoding='utf-8'))
    if prediction['schema_version'] != 1 or prediction['sampling_hz'] != 5:
        raise ValueError('Expected canonical schema v1 predictions at 5 Hz')
    sample_hz = prediction['sampling_hz']
    recording_start = prediction['recording_start']
    common_t0 = displayed_time(recording_start)
    calibration = Path(calibration)
    site = normalize_site(json.loads(calibration.read_text(encoding='utf-8')))
    plan = Path(plan) if plan is not None else calibration.with_name(site['plan'])
    source_dimensions = prediction.get('source_dimensions', {})
    image_sizes = {}
    grouped = defaultdict(list)
    for frame in prediction['frames']:
        grouped[frame['camera']].append(frame)
    if not grouped or grouped.keys() - media.keys():
        raise ValueError(f'Missing video inputs for cameras {sorted(grouped.keys() - media.keys())}')

    output = Path(output).resolve()
    cameras, frames, observations = [], [], []
    adjusted_frames = 0
    for camera, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row['pts'])
        dimensions = source_dimensions.get(str(camera), site.get('native_wh'))
        if dimensions is None:
            source_metadata = media_metadata(media[camera])
            if source_metadata is None:
                raise ValueError(f'Camera {camera}: source dimensions are required')
            dimensions = source_metadata[:2]
        width, height = dimensions
        if min(width, height) <= 0:
            raise ValueError(f'Camera {camera}: invalid source dimensions')
        image_sizes[camera] = (height, width)
        offset = (media_offsets or {}).get(camera)
        video, origin, metadata = prepare_media(media[camera], output / 'videos', camera, offset)
        prepared = [browser_frame(row, recording_start, origin) for row in rows]
        adjusted_frames += disambiguate_clock([frame for frame, _ in prepared], (camera,))
        for frame, detections in prepared:
            frames.append(frame)
            for detection in detections:
                detection[1] = frame[1]
            observations.extend(detections)
        video_width, video_height, duration_ms = metadata or (width, height, round(
            (rows[-1]['pts'] - origin + 1 / sample_hz) * 1000
        ))
        cameras.append({
            'camera_id': camera, 'plan_label': str(camera),
            'video': video.relative_to(output).as_posix(),
            'video_width': video_width, 'video_height': video_height,
            'bbox_width': width, 'bbox_height': height, 'video_duration_ms': duration_ms,
            'first_source_pts_ms': round(origin * 1000), 'display_mode': 'video',
        })

    tracks, intervals = summarize_tracks(observations, sample_hz)
    manifest = {
        'schema_version': 1, 'release_version': 'lol-predictions-v1', 'source_format': 'predictions',
        'profile': prediction['profile'], 'sampling_hz': sample_hz, 'common_t0': common_t0,
        'spatial_available': True, 'cameras': cameras, 'frame_counts': {'predicted': len(frames)},
        'ui_clock_adjusted_frames': adjusted_frames,
        'source_clock_semantics': 'Shared recording clock, with native camera PTS retained for video playback.',
    }
    report = {'valid': True, 'format': 'predictions', 'frames': len(frames),
              'observations': len(observations), 'tracks': len(tracks)}
    dataset = {'manifest': manifest, 'frames': frames, 'observations': observations,
               'tracks': tracks, 'intervals': intervals, 'validation': report}
    floor = output / 'floor'
    floor.mkdir(parents=True, exist_ok=True)
    target = floor / 'plan.svg'
    if plan.resolve() != target.resolve():
        copyfile(plan, target)
    viewer_fields = ('geometry', 'plan_view_box', 'display')
    viewer_site = {key: site[key] for key in viewer_fields if key in site}
    if site.get('anchors', {}).get('plan'):
        viewer_site['anchors'] = {'plan': site['anchors']['plan']}
    write_json(floor / 'calibration.json', viewer_site)
    write_json(floor / 'tracks-floor.json', floor_tracks(
        observations, site, image_sizes, sample_hz, common_t0, smoothing_sigma,
    ))
    with (output / 'tracks.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['track_id', 'first_seen_s', 'last_seen_s', 'cameras', 'observation_count'])
        for track in tracks:
            writer.writerow([track['track_id'], track['first_seen_s'], track['last_seen_s'],
                             ' '.join(map(str, track['cameras'])), track['observation_count']])
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'dataset.json', dataset)
    return output


def browser_frame(frame, recording_start, media_origin):
    camera, index, pts = frame['camera'], frame['frame_index'], frame['pts']
    time = round(frame['time'] - recording_start, 6)
    video_time = round(pts - media_origin, 6)
    if video_time < -.001:
        raise ValueError(f'Camera {camera} frame {index} precedes the browser media start')
    video_time = max(0.0, video_time)
    timestamp = displayed_time(frame['time'])
    native_ms = round(pts * 1000)
    rows = [[camera, time, video_time, str(track['global_id']), *track['box'],
             'prediction', native_ms, time, str(track['id']), str(track['global_id']),
             index, native_ms, 'shared_clock', timestamp, track['score']] for track in frame['tracks']]
    return [camera, time, video_time, 'predicted', native_ms, time, 'shared_clock',
            native_ms, index, len(rows), None, timestamp, 'source_pts'], rows


def summarize_tracks(observations, sample_hz):
    groups = defaultdict(list)
    views = defaultdict(list)
    for row in observations:
        groups[row[3]].append(row)
        views[row[3], row[0]].append(row)
    intervals = []
    for (identity, camera), rows in sorted(views.items()):
        run = []
        for row in sorted(rows, key=lambda row: row[2]):
            if run and (row[2] - run[-1][2] > 1.5 / sample_hz or row[1] < run[-1][1]):
                intervals.append(visibility_interval(identity, camera, run))
                run = []
            run.append(row)
        intervals.append(visibility_interval(identity, camera, run))
    tracks = []
    for identity, rows in sorted(groups.items()):
        cameras = sorted({row[0] for row in rows})
        tracks.append({
            'track_id': identity, 'global_id': identity, 'source_global_id': None,
            'link_status': 'prediction', 'first_seen_s': min(row[1] for row in rows),
            'last_seen_s': max(row[1] for row in rows), 'time_available': True,
            'camera_count': len(cameras), 'cameras': cameras, 'observation_count': len(rows),
        })
    return tracks, intervals


def visibility_interval(identity, camera, rows):
    first, last = rows[0], rows[-1]
    return {'track_id': identity, 'camera_id': camera, 'first_seen_s': first[1],
            'last_seen_s': last[1], 'first_video_ms': round(first[2] * 1000),
            'last_video_ms': round(last[2] * 1000), 'observation_count': len(rows)}


def displayed_time(value):
    # The pipeline stores the recording's displayed clock on a UTC numeric axis.
    return datetime.fromtimestamp(value, timezone.utc).replace(tzinfo=None).isoformat()


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False),
                    encoding='utf-8')
