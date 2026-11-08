"""Image-to-plan projections and measured geometric association evidence."""
import math
from bisect import bisect_left
from collections import defaultdict

import cv2
import numpy as np

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


def project(point, matrix):
    """Return planar coordinates and a local scale, measured via the Jacobian."""
    _, x, y, height = point
    value = matrix @ np.array([x, y, 1.])
    if abs(value[2]) < 1e-8:
        return None
    xy = value[:2] / value[2]
    jacobian = (matrix[:2, :2] - xy[:, None] * matrix[2, :2]) / value[2]
    scale = height * np.linalg.norm(jacobian) / np.sqrt(2.)
    return (xy, float(scale)) if np.isfinite(xy).all() and 1 <= scale < 4000 else None


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



def footpoint(timestamp, box, image_size):
    height, width = image_size
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    sy, sx = height / 1944, width / 2592
    if box_height >= 45 * sy and 30 * sy < y2 < 1910 * sy and 15 * sx < x1 < x2 < 2577 * sx:
        return timestamp, (x1 + x2) / 2, y2, box_height
    return None


def correct_position(xy, config):
    if config is None:
        return xy
    params = np.asarray(config['parameters'])
    if config['type'] == 'gaussian_affine':
        normalized = (xy - config['center']) / config['width']
        strength = np.exp(-.5 * np.sum(normalized ** 2, axis=-1, keepdims=True))
        offset, slope = params.reshape(2, 2).T
        return xy + strength * (offset + slope * normalized)
    if config['type'] == 'lateral_spline':
        y = xy[..., 1]
        lateral = params[0] * np.exp(-.5 * ((y - config['center']) / config['width']) ** 2)
        longitudinal = np.interp(y, config['knots'], np.r_[0., params[1:], 0.])
        return xy + np.stack((lateral, longitudinal), axis=-1)
    raise ValueError(f"Unsupported floor correction: {config['type']!r}")


def planar_tracks(frames, geometry, image_sizes):
    matrices = {int(camera): np.asarray(matrix) for camera, matrix in geometry.get('matrices', {}).items()}
    masks = {int(camera): [np.asarray(polygon, np.float32) for polygon in polygons]
             for camera, polygons in geometry.get('occlusions', {}).items()}
    corrections = {int(camera): config for camera, config in geometry.get('corrections', {}).items()}
    floor = np.asarray(geometry['walkable'], np.float32) if geometry.get('walkable') else None
    points = defaultdict(list)
    for frame in frames:
        camera = frame['camera']
        if camera not in matrices:
            continue
        for track in frame['tracks']:
            point = footpoint(frame['time'], track['box'], image_sizes[camera])
            if point is None or any(cv2.pointPolygonTest(mask, point[1:3], False) >= 0
                                    for mask in masks.get(camera, ())):
                continue
            value = project(point, matrices[camera])
            if value is None:
                continue
            xy = correct_position(value[0], corrections.get(camera))
            if floor is None or cv2.pointPolygonTest(floor, tuple(xy), True) >= -8:
                points[track['id']].append((frame['time'], xy))
    return {local: sorted(rows, key=lambda point: point[0]) for local, rows in points.items()}


def geometry_evidence(tracks, frames, raw, geometry, image_sizes):
    pairs = {tuple(sorted(pair)) for pair in geometry.get('overlap_pairs', ())}
    if not pairs:
        return [], set()
    points = planar_tracks(frames, geometry, image_sizes)
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
