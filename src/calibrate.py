"""Fit office floor homographies from fixed landmarks in native camera pixels."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def homography(landmarks):
    """Return the native-image-to-plan homography and fit residuals."""
    if len(landmarks) < 4:
        raise ValueError('at least four landmark pairs are required')
    image = np.asarray([point['image'] for point in landmarks], dtype=float)
    plan = np.asarray([point['plan'] for point in landmarks], dtype=float)
    for name, points in (('image', image), ('plan', plan)):
        if points.shape != (len(landmarks), 2) or not np.isfinite(points).all():
            raise ValueError(f'{name} must have two finite coordinates per landmark')
        if len(np.unique(points, axis=0)) != len(landmarks):
            raise ValueError(f'{name} landmark coordinates must be distinct')
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError(f'{name} landmarks are collinear')

    matrix, _ = cv2.findHomography(image, plan, method=0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError('landmark pairs do not determine an invertible homography')
    projected = np.column_stack((image, np.ones(len(image)))) @ matrix.T
    denominators = projected[:, 2]
    if np.any(np.abs(denominators) < 1e-8) or np.any(np.sign(denominators) != np.sign(denominators[0])):
        raise ValueError('homography has a horizon inside the landmark region')
    errors = np.linalg.norm(projected[:, :2] / denominators[:, None] - plan, axis=1)
    return matrix.tolist(), {
        'landmark_count': len(landmarks),
        'rmse_plan_units': float(np.sqrt(np.mean(errors ** 2))),
        'max_error_plan_units': float(errors.max()),
        'errors_plan_units': errors.tolist(),
    }


def recalibrate(site, landmarks, max_plan_error=None):
    """Update floor matrices and discard older fits involving the updated cameras."""
    if not landmarks:
        raise ValueError('no cameras with static landmarks were provided')
    results, matrices = {}, {}
    for camera, pairs in sorted(landmarks.items()):
        if not camera.isdecimal() or int(camera) < 1 or camera != str(int(camera)):
            raise ValueError(f'invalid camera ID {camera!r}')
        try:
            matrix, fit = homography(pairs)
            if max_plan_error is not None and fit['max_error_plan_units'] > max_plan_error:
                raise ValueError(f"maximum fit error {fit['max_error_plan_units']:.3g} exceeds {max_plan_error:g} plan units")
        except (KeyError, TypeError, ValueError, cv2.error) as error:
            raise ValueError(f'camera {camera}: {error}') from error
        matrices[camera] = matrix
        results[camera] = fit

    if 'office_geometry' in site:
        geometry = site['office_geometry']
        corrections = 'junction_corrections'
    else:
        geometry = site.setdefault('geometry', {})
        corrections = 'corrections'
    geometry.setdefault('matrices', {}).update(matrices)
    for camera in matrices:
        geometry.get(corrections, {}).pop(camera, None)
    disabled_pairs = set()
    for pairs in (site.get('pairwise', {}), site.get('association', {}).get('pair_geometry', {})):
        for pair in list(pairs):
            if matrices.keys() & set(pair.split('-')):
                del pairs[pair]
                disabled_pairs.add(pair)
    site['static_landmark_calibration'] = {
        'source': 'fixed floor landmarks in native image pixels',
        'fits': results,
        'disabled_legacy_camera_pairs': sorted(disabled_pairs),
    }
    return site


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks', type=Path, required=True,
                        help='JSON {"landmarks": {"2": [{"image": [x, y], "plan": [u, v]}, ...]}}')
    parser.add_argument('--site', type=Path, required=True, help='existing pipeline site or viewer calibration JSON')
    parser.add_argument('--output', type=Path, required=True, help='path for a separate updated JSON file')
    parser.add_argument('--max-plan-error', type=float, help='reject any fit exceeding this error on its input points')
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.site.resolve(), args.landmarks.resolve()}:
        parser.error('--output must differ from both input files')
    if args.max_plan_error is not None and (not np.isfinite(args.max_plan_error) or args.max_plan_error <= 0):
        parser.error('--max-plan-error must be finite and positive')
    try:
        site = recalibrate(read_json(args.site), read_json(args.landmarks)['landmarks'], args.max_plan_error)
        args.output.write_text(json.dumps(site, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
