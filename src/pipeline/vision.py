"""Video sampling, YOLO/ByteTrack replay and exact CLIP-ReID crop extraction."""
from collections import defaultdict
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .graph import MAX_GAP
from .scene import box_overlap

SAMPLING_HZ = 5
TRACKER = {
    'track_buffer': 10, 'track_high_thresh': .25, 'track_low_thresh': .10,
    'new_track_thresh': .25, 'match_thresh': .8, 'fuse_score': True,
}


def sample_video(path, camera, settings, recording_start, recording_end):
    import av

    clock = settings.get('clock_anchors')
    offset = settings.get('offset_s', 0.)
    if clock:
        anchor_pts, anchor_time = np.asarray(clock, dtype=float).T
        if np.any(np.diff(anchor_pts) <= 0) or np.any(np.diff(anchor_time) < 0):
            raise ValueError(f'Camera {camera}: invalid clock anchor order')
    elif recording_start is None:
        raise ValueError(f'Camera {camera}: recording_start or clock_anchors are required')

    with av.open(str(path)) as source:
        stream = source.streams.video[0]
        stream.thread_type = 'AUTO'
        tick = 0
        for frame_index, frame in enumerate(source.decode(stream)):
            if frame.pts is None:
                continue
            pts = float(frame.pts * frame.time_base)
            if pts + 1e-7 < tick / SAMPLING_HZ:
                continue
            while tick / SAMPLING_HZ < pts - .12 - 1e-7:
                tick += 1
            if pts + 1e-7 < tick / SAMPLING_HZ:
                continue
            if clock:
                wall = float(np.interp(pts, anchor_pts, anchor_time))
                wall += pts - float(np.clip(pts, anchor_pts[0], anchor_pts[-1]))
            else:
                wall = recording_start + offset + pts
            if recording_end is not None and wall >= recording_end:
                break
            if recording_start is None or wall >= recording_start:
                yield {'camera': camera, 'frame_index': frame_index, 'pts': pts, 'time': wall, 'tick': tick}, frame
            tick += 1


class CameraTracker:
    """Replay actual detection rows; a predicted Kalman box is never exported."""

    def __init__(self, camera, image_size, options):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.camera = camera
        self.options = SimpleNamespace(**options)
        self.tracker = BYTETracker(self.options)
        self.tracker_factory = BYTETracker
        self.boxes_factory = Boxes
        self.image_size = tuple(image_size)
        self.segment = 0
        self.previous_tick = None

    def update(self, tick, detections, *, reset=False):
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        if self.previous_tick is not None:
            missing = tick - self.previous_tick - 1
            if reset or missing >= self.options.track_buffer:
                self.tracker = self.tracker_factory(self.options)
                self.segment += 1
            else:
                empty = self.boxes_factory(np.empty((0, 6), dtype=np.float32), self.image_size)
                for _ in range(missing):
                    self.tracker.update(empty)
        self.previous_tick = tick
        boxes = self.boxes_factory(detections, self.image_size)
        tracks = []
        for *_, raw_id, score, _cls, detection_index in self.tracker.update(boxes):
            box = detections[int(detection_index), :4].tolist()
            tracks.append({'id': f'{self.camera}:{self.segment}:{int(raw_id)}', 'box': box, 'score': float(score)})
        return tracks


def track_camera(cached, camera, screens):
    config = cached['config']
    tracker = CameraTracker(camera, config['image_size'], TRACKER | config.get('tracker', {}))
    frames = []
    previous = None
    for row in cached['frames']:
        tick = row['tick']
        point = (row['frame_index'], tick, row['pts'], row['time'])
        if row['camera'] != camera or (previous and
                (any(a >= b for a, b in zip(previous[:3], point[:3], strict=True))
                 or previous[3] > point[3])):
            raise ValueError(f'Camera {camera}: cache frames must have increasing source indexes, ticks and PTS')
        if 'detections' not in row:
            raise ValueError(f"Camera {camera} frame {row['frame_index']}: missing raw detections")
        reset = previous is not None and row['time'] - previous[3] > MAX_GAP
        previous = point
        detections = row['detections']
        if screens and len(detections):
            inside = box_overlap([d[:4] for d in detections], screens).max(axis=1) >= .70
            detections = [det for det, covered in zip(detections, inside, strict=True) if not covered]
        frame = {key: row[key] for key in ('camera', 'frame_index', 'pts', 'time', 'tick')}
        frames.append({**frame, 'tracks': tracker.update(tick, detections, reset=reset)})
    return frames


def local_model(model_dir, name):
    if model_dir is None:
        raise ValueError(f'Model directory required to load {name}')
    path = Path(model_dir) / name
    if not path.exists():
        raise FileNotFoundError(f'Local model is missing: {path}')
    return path


def load_yolo(model_dir):
    from ultralytics import YOLO

    return YOLO(str(local_model(model_dir, 'yolo11m.pt')))


def detect_camera(model, path, camera, settings, recording_start, recording_end, device, progress):
    samples = sample_video(path, camera, settings, recording_start, recording_end)
    frames, image_size = [], None
    while batch := list(islice(samples, 8)):
        images = [frame.to_ndarray(format='bgr24') for _, frame in batch]
        if image_size is None:
            image_size = list(images[0].shape[:2])
        detections = model.predict(images, device=device, imgsz=960,
                                   conf=TRACKER['track_low_thresh'], classes=[0],
                                   verbose=False, save=False)
        for (metadata, _), result in zip(batch, detections, strict=True):
            frames.append({**metadata, 'detections': result.boxes.cpu().numpy().data.tolist()})
        if len(frames) % 200 == 0:
            progress(f'Camera {camera}: detected people in {len(frames)} sampled frames')
    if image_size is None:
        raise ValueError(f'Camera {camera}: no sampled frames in {path}')
    config = {'image_size': image_size, 'tracker': TRACKER, 'fps': SAMPLING_HZ}
    if recording_start is not None:
        config['start'] = recording_start
    if recording_end is not None:
        config['end'] = recording_end
    return {'config': config, 'frames': frames}


def frames_at_pts(path, timestamps):
    """Decode selected frames by exact stream PTS, reusing the decoder over short gaps."""
    import av

    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_count = 2
        decoder, last_pts = None, None
        for pts in sorted(set(timestamps)):
            stamp = round(pts / float(stream.time_base))
            if last_pts is None or pts - last_pts > 1.:
                video.seek(stamp, stream=stream, backward=True)
                decoder = video.decode(stream)
            frame = next((frame for frame in decoder if frame.pts is not None and frame.pts >= stamp), None)
            if frame is None or frame.pts != stamp:
                raise ValueError(f'Expected video PTS {pts:.6f}s in {path}')
            last_pts = pts
            yield pts, frame.to_image()


class Appearance:
    """Store by exact camera, source frame and rounded box, then fill cache misses from video."""

    def __init__(self, known, videos, pts, model_dir, device, progress):
        self.known = known
        self.videos = videos
        self.pts = pts
        self.model_dir = model_dir
        self.device = device
        self.encoder = None
        self.progress = progress

    def ensure(self, crops):
        missing = crops - self.known.keys()
        if not missing:
            return
        without_video = sorted({crop[0] for crop in missing} - self.videos.keys())
        if without_video:
            example = next(crop for crop in sorted(missing) if crop[0] in without_video)
            raise ValueError(f'Appearance cache is missing {len(missing)} required crop(s). '
                             f'Provide source video for cameras {without_video}; example {example}.')
        import torch
        from PIL import Image
        from torch.nn.functional import normalize

        if self.encoder is None:
            from .clip_vision import VisionTransformer

            checkpoint = local_model(self.model_dir, 'clip-reid-msmt17.pth')
            model = VisionTransformer(16, 8, 16, 16, 768, 12, 12, 512)
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            prefix = 'image_encoder.'
            model.load_state_dict({key.removeprefix(prefix): value for key, value in state.items()
                                   if key.startswith(prefix)})
            self.encoder = model.eval().to(self.device)

        by_camera = defaultdict(lambda: defaultdict(list))
        for crop in sorted(missing):
            by_camera[crop[0]][self.pts[crop[:2]]].append(crop)
        batch, keys = [], []

        @torch.inference_mode()
        def flush():
            if not batch:
                return
            tensor = torch.from_numpy(np.stack(batch)).to(device=self.device, dtype=torch.float32)
            _, hidden, projected = self.encoder((tensor.permute(0, 3, 1, 2) / 255 - .5) / .5)
            vectors = normalize(torch.cat((hidden[:, 0], projected[:, 0]), dim=1), dim=1).cpu().tolist()
            self.known.update(zip(keys, vectors, strict=True))
            keys.clear()
            batch.clear()

        for camera, requests in sorted(by_camera.items()):
            self.progress(f'Camera {camera}: encoding {sum(map(len, requests.values()))} CLIP crops')
            for pts, image in frames_at_pts(self.videos[camera], requests):
                for crop in requests[pts]:
                    x1, y1, x2, y2 = crop[2]
                    box = max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f'Invalid CLIP crop: {crop}')
                    batch.append(np.asarray(image.crop(box).resize((128, 256), Image.Resampling.BILINEAR)))
                    keys.append(crop)
                    if len(batch) == 16:
                        flush()
            flush()
