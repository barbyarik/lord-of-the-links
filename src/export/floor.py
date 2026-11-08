"""Project tracked boxes onto the configured floor and fuse simultaneous views."""
from collections import defaultdict
from itertools import combinations
from math import ceil, cos, exp, hypot, radians
from statistics import median

from ..floor_geometry import FloorProjection


def distance_to_polygon(point, polygon):
    """Signed distance, positive inside. Coordinates are in plan units."""
    x, y = point
    inside = False
    distance = float('inf')
    for (ax, ay), (bx, by) in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if not length_squared:
            continue
        fraction = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_squared))
        distance = min(distance, hypot(x - ax - fraction * dx, y - ay - fraction * dy))
        if (ay > y) != (by > y) and x < ax + (y - ay) * dx / dy:
            inside = not inside
    return distance if inside else -distance


def floor_tracks(observations, site, image_sizes, sample_hz, common_t0, smoothing_sigma):
    """Fuse accepted observations per global ID on the shared sampled grid."""
    geometry = site['geometry']
    projection = FloorProjection(geometry, image_sizes, site.get('guards'))
    walkable = geometry.get('walkable', ())
    points = defaultdict(list)
    for row in observations:
        camera, time, video_time, identity, *box = row[:8]
        projected = projection.point(camera, box)
        if projected is None:
            continue
        tick = round(time * sample_hz)
        points[identity, tick].append((
            box[3] - box[1],
            {'camera_id': camera, 'frame_index': row[13], 'video_ms': round(video_time * 1000),
             't_s': time, 'local_id': row[11], 'x': projected[0], 'y': projected[1]},
        ))

    tracks = defaultdict(list)
    radius = projection.guards['fusion_diameter_plan_units']
    for (identity, tick), observations_at_tick in sorted(points.items()):
        time = tick / sample_hz
        by_camera = {}
        for height, point in sorted(observations_at_tick, key=lambda item: abs(item[1]['t_s'] - time)):
            by_camera.setdefault(point['camera_id'], (height, point))
        views = list(by_camera.values())
        sources = [point for _, point in views]
        diameter = max((hypot(a['x'] - b['x'], a['y'] - b['y'])
                        for a, b in combinations(sources, 2)), default=0.0)
        if diameter <= radius:
            x = median(point['x'] for point in sources)
            y = median(point['y'] for point in sources)
            mode = 'median' if len(sources) > 1 else 'single_camera'
        else:
            _, best = max(views, key=lambda item: item[0])
            x, y, mode = best['x'], best['y'], 'largest_box_disagreement'
        tracks[identity].append({'t_s': time, 'x': x, 'y': y, 'mode': mode,
                                 'diameter': round(diameter, 3), 'sources': sources})
    for rows in tracks.values():
        for segment, run in enumerate(continuous_runs(rows, sample_hz)):
            for point in run:
                point['segment'] = segment
                point['x_raw'], point['y_raw'] = point['x'], point['y']
            if smoothing_sigma > 0 and len(run) >= 5:
                smooth_positions(run, smoothing_sigma * sample_hz, walkable)
    return {'schema_version': 1, 'sample_hz': sample_hz, 'displayed_clock_zero': common_t0,
            'plan': 'plan.svg', 'smoothing_sigma_s': smoothing_sigma, 'tracks': dict(tracks)}


def continuous_runs(rows, sample_hz):
    start = 0
    for index, (previous, current) in enumerate(zip(rows, rows[1:]), start=1):
        if current['t_s'] - previous['t_s'] > 1.01 / sample_hz or hypot(
                current['x'] - previous['x'], current['y'] - previous['y']) > 65:
            yield rows[start:index]
            start = index
    yield rows[start:]


def smooth_positions(rows, sigma_samples, walkable):
    """Apply Gaussian smoothing while preserving bends, endpoints and wall clearance."""
    raw = [(point['x'], point['y']) for point in rows]
    result = raw.copy()
    locked = protected_corners(raw)
    radius = int(4 * sigma_samples + .5)
    weights = [exp(-.5 * (step / sigma_samples) ** 2) for step in range(-radius, radius + 1)]
    total = sum(weights)
    clearance = [link_depth(a, b, walkable) for a, b in zip(raw, raw[1:])]
    for index in range(1, len(rows) - 1):
        if index in locked:
            continue
        neighbors = [raw[max(0, min(len(raw) - 1, index + step))]
                     for step in range(-radius, radius + 1)]
        x, y = raw[index]
        dx = sum(weight * point[0] for weight, point in zip(weights, neighbors)) / total - x
        dy = sum(weight * point[1] for weight, point in zip(weights, neighbors)) / total - y
        scale = min(1.0, 10.0 / max(hypot(dx, dy), 1e-9))
        depth = max(0, -distance_to_polygon((x, y), walkable))
        for fraction in (scale, scale / 2, scale / 4):
            candidate = x + dx * fraction, y + dy * fraction
            if (distance_to_polygon(candidate, walkable) >= -depth - 1e-6
                    and link_depth(result[index - 1], candidate, walkable) <= clearance[index - 1] + 1e-6
                    and link_depth(candidate, result[index + 1], walkable) <= clearance[index] + 1e-6):
                result[index] = candidate
                rows[index]['x'], rows[index]['y'] = candidate
                break


def link_depth(left, right, polygon):
    dx, dy = right[0] - left[0], right[1] - left[1]
    steps = max(1, ceil(hypot(dx, dy)))
    depth = min(distance_to_polygon((left[0] + dx * i / steps, left[1] + dy * i / steps), polygon)
                for i in range(steps + 1))
    return max(0.0, -depth)


def protected_corners(raw):
    candidates = []
    for i in range(3, len(raw) - 3):
        x, y = raw[i]
        left_x, left_y = raw[i - 3]
        right_x, right_y = raw[i - 2]
        before = x - (left_x + right_x) / 2, y - (left_y + right_y) / 2
        left_x, left_y = raw[i + 2]
        right_x, right_y = raw[i + 3]
        after = (left_x + right_x) / 2 - x, (left_y + right_y) / 2 - y
        lengths = hypot(*before), hypot(*after)
        if min(lengths) < 8:
            continue
        alignment = (before[0] * after[0] + before[1] * after[1]) / (lengths[0] * lengths[1])
        if cos(radians(125)) <= alignment <= cos(radians(65)):
            candidates.append((abs(alignment), i))
    locked = set()
    for _, index in sorted(candidates):
        if all(abs(index - old) > 4 for old in locked):
            locked.add(index)
    return locked
