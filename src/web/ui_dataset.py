"""Build the UI projection without changing source records or source segments."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta

from .dataset_io import time_value
from ..export.timing import disambiguate_clock


def camera_manifest(data, origin, media):
    """Choose actual video files or source JPEG previews; never transcode implicitly."""
    if data['kind'] == 'handoff':
        manifest = deepcopy(data['manifest'])
    else:
        manifest = {
            'release_version': data['export']['release_version'],
            'common_t0': origin.isoformat() if origin else None,
            'sampling_hz': 2,
            'cameras': [],
        }
        for camera_id, camera in data['cameras'].items():
            source = (data['camera_root'] / camera['source']).resolve()
            playable = source.is_file() and source.suffix.lower() in {'.mp4', '.webm'}
            video_url = f'/media/video/{camera_id}'
            if playable:
                media[video_url] = str(source)
            manifest['cameras'].append({
                'camera_id': camera_id,
                'plan_label': camera['plan_label'],
                'video': video_url if playable else None,
                'video_width': camera['width'],
                'video_height': camera['height'],
                'bbox_width': camera['width'],
                'bbox_height': camera['height'],
                'video_duration_ms': camera['container_duration_s'] * 1000,
                'display_mode': 'video' if playable else 'images',
            })
    for camera in manifest['cameras']:
        camera.setdefault('display_mode', 'video')
    manifest['source_format'] = data['kind']
    manifest['spatial_available'] = data['kind'] == 'handoff'
    manifest['source_clock_semantics'] = (
        'Timezone-free displayed/estimated clock; not calibrated synchronization.'
    )
    return manifest


def project_frames(data, origin, media):
    def seconds(value):
        clock = time_value(value)
        if clock is None or origin is None:
            return None
        return (clock - origin).total_seconds()

    anchors = defaultdict(list)
    for frame in data['frames']:
        time = seconds(frame.get('clock_time'))
        if time is not None and frame['pts_ms'] is not None:
            anchors[frame['camera_id']].append((frame['pts_ms'] / 1000, time))

    frames = []
    for frame in data['frames']:
        camera_id = frame['camera_id']
        source_time = seconds(frame.get('clock_time'))
        time = source_time
        timing = 'source_clock'
        if time is None and anchors[camera_id]:
            point = (frame['pts_ms'] if frame['pts_ms'] is not None else frame['sample_ms']) / 1000
            anchor = min(anchors[camera_id], key=lambda item: abs(item[0] - point))
            time = anchor[1] + point - anchor[0]
            timing = 'ui_estimated_from_pts'
        if time is None:
            timing = 'unmapped'
        if frame.get('video_ms') is not None:
            video_time = float(frame['video_ms']) / 1000
        else:
            video_time = frame['pts_ms'] / 1000 if frame['pts_ms'] is not None else None
        image_url = None
        if frame.get('image'):
            image_url = f'/media/frame/{camera_id}/{frame["sample_ms"]}'
            media[image_url] = str((data['annotation_root'] / frame['image']).resolve())
        frames.append([
            camera_id, time, video_time, frame['review_status'], frame['sample_ms'],
            source_time, frame['clock_status'], frame['pts_ms'], frame['frame_index'],
            frame['person_count'], image_url, frame.get('clock_time'), timing,
        ])
    return frames


def build_ui(data):
    legacy = data['kind'] == 'handoff'
    clocks = [time_value(frame['clock_time']) for frame in data['frames'] if frame.get('clock_time')]
    origin = time_value(data['manifest']['common_t0']) if legacy else min(clocks, default=None)
    media = {}
    frames = project_frames(data, origin, media)
    # Include leading samples whose clock was unreadable/missing. This changes
    # only the relative UI origin; ISO clock values stay in the source records.
    shift = min((row[1] for row in frames if row[1] is not None), default=0)
    if not legacy and shift < 0:
        origin += timedelta(seconds=shift)
        for row in frames:
            for column in (1, 5):
                if row[column] is not None:
                    row[column] -= shift
    manifest = camera_manifest(data, origin, media)
    manifest['ui_clock_adjusted_frames'] = disambiguate_clock(frames, data['cameras'])
    manifest['ui_origin_estimated'] = not legacy and shift < 0
    index = {(row[0], row[4]): row for row in frames}
    source_tracks = {track['id']: track for track in data['export']['tracks']}
    observations = []
    groups = defaultdict(list)
    source_global_ids = {}
    for observation in data['observations']:
        identity = observation['track_id'] if legacy else observation.get('global_id') or observation['local_id']
        frame = index[observation['camera_id'], observation['sample_ms']]
        row = [
            observation['camera_id'], frame[1], frame[2], identity,
            *[observation[key] for key in ('x_min', 'y_min', 'x_max', 'y_max')],
            source_tracks[identity]['link_status'], observation['sample_ms'], frame[5],
            observation['local_id'], observation.get('global_id'), observation['frame_index'],
            observation['pts_ms'], observation['clock_status'], observation.get('clock_time'),
        ]
        observations.append(row)
        groups[identity].append(row)
        source_global_ids.setdefault(identity, observation.get('source_global_id'))

    tracks, intervals = [], []
    for identity, track in source_tracks.items():
        mapped_times = [row[1] for row in groups[identity] if row[1] is not None]
        cameras = sorted({row[0] for row in groups[identity]})
        tracks.append({
            'track_id': identity,
            'source_global_id': source_global_ids[identity] if legacy else None,
            'global_id': next((row[12] for row in groups[identity] if row[12]), None),
            'link_status': track['link_status'],
            'first_seen_s': min(mapped_times) if mapped_times else None,
            'last_seen_s': max(mapped_times) if mapped_times else None,
            'time_available': bool(mapped_times),
            'camera_count': len(cameras), 'cameras': cameras,
            'observation_count': len(groups[identity]),
            'segments': deepcopy(track['segments']),
        })
        for segment in track['segments']:
            first = index[segment['camera_id'], segment['first_sample_ms']]
            last = index[segment['camera_id'], segment['last_sample_ms']]
            intervals.append({
                'track_id': identity, 'camera_id': segment['camera_id'],
                'local_id': segment['local_id'],
                'first_seen_s': first[1], 'last_seen_s': last[1],
                'first_video_ms': first[2] * 1000 if first[2] is not None else None,
                'last_video_ms': last[2] * 1000 if last[2] is not None else None,
                'source_segment': deepcopy(segment),
            })

    manifest['frame_counts'] = dict(Counter(frame['review_status'] for frame in data['frames']))
    report = {
        'valid': True, 'format': data['kind'], 'release_version': data['export']['release_version'],
        'frames': len(frames), 'observations': len(observations), 'tracks': len(tracks),
        'frame_status_counts': manifest['frame_counts'],
        'unmapped_clock_frames': sum(row[1] is None for row in frames),
        'warnings': data['warnings'],
        'checks': [
            'keys', 'full_grid' if not legacy else 'handoff_grid', 'frame_status', 'pts_window',
            'observation_frame_fields', 'bbox_bounds', 'person_count', 'source_segment_endpoints',
            'source_segment_continuity', 'segment_counts', 'complete_segment_coverage', 'metadata_counts',
        ],
    }
    return {
        'manifest': manifest, 'frames': frames, 'observations': observations,
        'tracks': tracks, 'intervals': intervals, 'validation': report,
        'source': {
            'frames': data['frames'], 'observations': data['observations'],
            'tracks': data['export'], 'cameras': data['camera_rows'], **data['reviews'],
        },
        '_media_files': media,
    }
