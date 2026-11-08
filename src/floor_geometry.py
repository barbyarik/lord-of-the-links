"""Shared floor calibration and native-pixel projection for stitching and export."""
import cv2
import numpy as np

DEFAULT_GUARDS = {
    'min_box_height_px': 45,
    'foot_y_px_exclusive': [30, 1910],
    'box_x_bounds_px_exclusive': [15, 2577],
    'local_scale_inclusive_exclusive': [1, 4000],
    'walkable_slack_plan_units': 8,
    'fusion_diameter_plan_units': 18.0,
}


def normalize_site(site):
    """Translate archived viewer-only calibration files into the current geometry format."""
    if 'geometry' in site or 'office_geometry' not in site:
        return site
    old = site['office_geometry']
    scale = site['anchors'].get('image_scale', 3)
    masks = old.get('occlusions')
    if masks is None:
        masks = {camera: [(np.asarray(polygon) * scale).tolist() for polygon in polygons]
                 for camera, polygons in old.get('occlusion_polygons_in_864x648', {}).items()}
    corrections = {camera: correction for camera, params in old.get('junction_corrections', {}).items()
                   if (correction := legacy_correction(camera, params)) is not None}
    return {**site, 'geometry': {
        'matrices': old['matrices'], 'walkable': site['anchors']['walkable'],
        'occlusions': masks, 'corrections': corrections | old.get('corrections', {}),
    }}


def legacy_correction(camera, parameters):
    """Convert residuals from the historical office-map handoff."""
    if camera == '2':
        return {'type': 'gaussian_affine', 'center': (292, 185), 'width': (24, 26),
                'parameters': parameters}
    if camera == '5':
        return {'type': 'lateral_spline', 'center': 350, 'width': 65,
                'knots': (280, 320, 350, 380, 420, 480), 'parameters': parameters}
    return None


def footpoint(timestamp, box, image_size, guards=DEFAULT_GUARDS):
    height, width = image_size
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    sy, sx = height / 1944, width / 2592
    y_min, y_max = guards['foot_y_px_exclusive']
    x_min, x_max = guards['box_x_bounds_px_exclusive']
    if (box_height >= guards['min_box_height_px'] * sy and y_min * sy < y2 < y_max * sy
            and x_min * sx < x1 < x2 < x_max * sx):
        return timestamp, (x1 + x2) / 2, y2, box_height
    return None


def project(point, matrix, scale_limits):
    """Return planar coordinates and a local scale, measured via the Jacobian."""
    _, x, y, height = point
    value = matrix @ np.array([x, y, 1.])
    if abs(value[2]) < 1e-8:
        return None
    xy = value[:2] / value[2]
    jacobian = (matrix[:2, :2] - xy[:, None] * matrix[2, :2]) / value[2]
    scale = height * np.linalg.norm(jacobian) / np.sqrt(2.)
    return (xy, float(scale)) if np.isfinite(xy).all() and scale_limits[0] <= scale < scale_limits[1] else None


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


class FloorProjection:
    """Prepare one site against the actual native size of each input camera."""

    def __init__(self, geometry, image_sizes, guards=None):
        self.image_sizes = image_sizes
        self.guards = DEFAULT_GUARDS | (guards or {})
        self.matrices = {int(camera): np.asarray(matrix) for camera, matrix in geometry.get('matrices', {}).items()}
        self.masks = {int(camera): [np.asarray(polygon, np.float32) for polygon in polygons]
                      for camera, polygons in geometry.get('occlusions', {}).items()}
        self.corrections = {int(camera): config for camera, config in geometry.get('corrections', {}).items()}
        walkable = geometry.get('walkable')
        self.floor = np.asarray(walkable, np.float32) if walkable else None

    def point(self, camera, box):
        matrix = self.matrices.get(camera)
        if matrix is None:
            return None
        foot = footpoint(0, box, self.image_sizes[camera], self.guards)
        if foot is None or any(cv2.pointPolygonTest(mask, foot[1:3], False) >= 0
                               for mask in self.masks.get(camera, ())):
            return None
        projected = project(foot, matrix, self.guards['local_scale_inclusive_exclusive'])
        if projected is None:
            return None
        xy = correct_position(projected[0], self.corrections.get(camera))
        if (self.floor is None
                or cv2.pointPolygonTest(self.floor, tuple(xy), True) >= -self.guards['walkable_slack_plan_units']):
            return tuple(map(float, xy))
        return None
