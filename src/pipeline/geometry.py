"""Image-to-plan projections and measured geometric association evidence."""
import math
from bisect import bisect_left
from collections import defaultdict

import cv2
import numpy as np

from ..floor_geometry import FloorProjection, footpoint

TIME_TOLERANCE = .45
POSITION_SCALE = .65


def aligned_points(left, right):
    """Match each left point to at most one near-synchronous right point."""
    right_times = [p[0] for p in right]
    used = set()
    for point in left:
        i = bisect_left(right_times, point[0])
        available = [j for j in (i - 1, i) if 0 <= j < len(right) and j not in used]
        if available:
            j = min(available, key=lambda k: abs(right_times[k] - point[0]))
            if abs(right_times[j] - point[0]) <= TIME_TOLERANCE:
                used.add(j)
                yield point, right[j]


def residuals(pairs, matrix):
    left, right = (np.asarray(side, dtype=float) for side in zip(*pairs, strict=True))
    src, dst = left[:, 1:3], right[:, 1:3]
    # The denominator check prevents a point near the projective horizon from
    # being interpreted as a valid match when OpenCV maps it to zero.
    def project(points, transform):
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ transform.T
        valid = np.abs(homogeneous[:, 2]) > 1e-8
        result = np.full((len(points), 2), np.inf)
        result[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:]
        return result

    forward = np.linalg.norm(project(src, matrix) - dst, axis=1) / right[:, 3]
    backward = np.linalg.norm(project(dst, np.linalg.inv(matrix)) - src, axis=1) / left[:, 3]
    return (forward + backward) / 2


def edge_positions(tracks, edges, models, image_sizes):
    if not models:
        return {}
    points = {identity: [point for camera, _index, time, box in rows
                         if (point := footpoint(time, box, image_sizes[camera]))]
              for identity, rows in tracks.items()}
    prepared = {pair: (np.asarray(model['matrix']), [np.asarray(h, dtype=np.float32)
                                                   for h in model['floor_support_hulls']])
                for pair, model in models.items()}
    positions = {}
    for _score, left, right, gap in edges:
        ca, cb = tracks[left][0][0], tracks[right][0][0]
        key = f'{min(ca, cb)}-{max(ca, cb)}'
        if gap > 0 or key not in prepared:
            continue
        first, second = (left, right) if ca < cb else (right, left)
        matrix, (hull_a, hull_b) = prepared[key]
        aligned = []
        for a, b in aligned_points(points[first], points[second]):
            if (cv2.pointPolygonTest(hull_a, a[1:3], True) < -20
                    or cv2.pointPolygonTest(hull_b, b[1:3], True) < -20):
                continue
            if not aligned or a[0] - aligned[-1][0][0] >= .35:
                aligned.append((a, b))
        if len(aligned) >= 3:
            errors = residuals(aligned, matrix)
            finite = errors[np.isfinite(errors)]
            if len(finite) >= 3:
                error = float(np.median(finite))
                positions[left, right] = math.exp(-.5 * (error / POSITION_SCALE) ** 2), error, len(finite)
    return positions


def overlap(left, right):
    start, end = max(left[0][0],right[0][0]), min(left[-1][0],right[-1][0])
    if end - start < .6:
        return ()
    times = [t for t,_ in right]
    observed, prev, used = [], -np.inf, set()
    for time, point in left:
        if not start-.18 <= time <= end+.18 or time-prev < .27:
            continue
        at = bisect_left(times, time)
        close = [i for i in (at-1,at) if 0 <= i < len(right) and i not in used]
        if not close:
            continue
        i = min(close, key=lambda k: abs(times[k]-time))
        if abs(times[i]-time) <= .22:
            observed.append(float(np.linalg.norm(point-right[i][1])))
            prev = time
            used.add(i)
    return observed



def planar_tracks(frames, projection):
    points = defaultdict(list)
    for frame in frames:
        camera = frame['camera']
        for track in frame['tracks']:
            xy = projection.point(camera, track['box'])
            if xy is not None:
                points[track['id']].append((frame['time'], np.asarray(xy)))
    return {local: sorted(rows, key=lambda point: point[0]) for local, rows in points.items()}


def geometry_evidence(tracks, frames, raw, geometry, image_sizes, guards=None):
    pairs = {tuple(sorted(pair)) for pair in geometry.get('overlap_pairs', ())}
    if not pairs:
        return [], set()
    points = planar_tracks(frames, FloorProjection(geometry, image_sizes, guards))
    simultaneous, far = [], set()
    for appearance, a, b, _ in raw:
        ca, cb = tracks[a][0][0], tracks[b][0][0]
        if ca == cb or tuple(sorted((ca, cb))) not in pairs or a not in points or b not in points:
            continue
        left, right = (a, b) if tracks[a][0][2] <= tracks[b][0][2] else (b, a)
        delta = tracks[right][0][2] - tracks[left][-1][2]
        if delta >= 0:
            continue
        errors = overlap(points[left], points[right])
        if len(errors) < 3:
            continue
        median, p80 = np.quantile(errors, [.5, .8]).tolist()
        fraction = sum(distance <= 18 for distance in errors) / len(errors)
        if len(errors) >= 5 and median >= 54 and min(errors) >= 30:
            far.add(tuple(sorted((a, b))))
        simultaneous.append(dict(a=a, b=b, gap=delta, app=appearance, median=median,
                                 p80=p80, near=fraction, n=len(errors)))
    return simultaneous, far


def strict_conflicts(simultaneous, far):
    return {tuple(sorted((row['a'], row['b']))) for row in simultaneous
            if row['median'] >= 100 and row['n'] >= 7 and row['near'] == 0
            and tuple(sorted((row['a'], row['b']))) in far}
