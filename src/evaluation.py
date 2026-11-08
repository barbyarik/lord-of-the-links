"""Exact-source-frame evaluation of MTMC predictions against the 2fps reference.

Association scores below are *conditional on IoU-matched observations*, not
IDF1/HOTA and not end-to-end recall. Detection misses are reported separately.
Naive reference clock strings are placed on a synthetic UTC epoch solely to
match the caller's time representation; this does not establish real UTC or
calibrated camera synchronization. This module never writes reference files.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from itertools import combinations
import math
from pathlib import Path

import lap
import numpy as np
from trackeval.metrics import HOTA, Identity

IOU_THRESHOLD = 0.5
MIN_BINS = (2, 3, 5, 8, 10, 15)
BIN_HZ = 5


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _clock(value: str) -> float:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _box(value) -> list[float]:
    box = [float(x) for x in value]
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError(f"Invalid xyxy box: {value!r}")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"Non-positive xyxy box: {value!r}")
    return box


def _iou_matrix(reference: list[list[float]], predicted: list[list[float]]) -> np.ndarray:
    a = np.asarray(reference, dtype=float).reshape(-1, 4)
    b = np.asarray(predicted, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=float)
    lower = np.maximum(a[:, None, :2], b[None, :, :2])
    upper = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection)


def _match(reference: list[list[float]], predicted: list[list[float]]) -> list[tuple[int, int, float]]:
    iou = _iou_matrix(reference, predicted)
    n, m = iou.shape
    if not n or not m:
        return []
    # One dummy column per GT allows every reference object to stay unmatched.
    # The bonus makes cardinality dominate IoU, then maximizes summed IoU.
    bonus = min(n, m) + 1
    cost = np.zeros((n, m + n), dtype=float)
    cost[:, :m] = np.where(iou >= IOU_THRESHOLD, -bonus - iou, 1e6)
    _, columns, _ = lap.lapjv(cost, extend_cost=True)
    return [(i, int(j), float(iou[i, j])) for i, j in enumerate(columns)
            if 0 <= j < m and iou[i, j] >= IOU_THRESHOLD]


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def _cross_pairs(counts_by_camera: dict[int, int]) -> int:
    counts = list(counts_by_camera.values())
    total = sum(counts)
    return (total * total - sum(n * n for n in counts)) // 2


def evaluation_data(root: Path, frames: list[dict], start: float, end: float,
                    max_gap_s: float, exclude_window: tuple[float, float] | None = None,
                    include_standard_frames: bool = False) -> dict:
    """Prepare fixed matches once, for many calls to :func:`measure`.

    ``root`` is the project root, or its ``dataset-2fps`` directory. ``start``
    and ``end`` are synthetic epoch seconds on the reference screen-clock axis;
    the evaluation window is [start, end). Input frames use camera, frame_index,
    time and tracks[{id, box, score}]. IDs must be camera-scoped local node IDs.
    ``exclude_window`` optionally removes a half-open interval from scoring.

    Reference episodes are derived over the ENTIRE reference before applying
    the window, split only when the gap exceeds max_gap_s. Set max_gap_s=90 for
    a 90-second episode definition. Unknown identities and unresolved group
    relations are excluded from strict association, but retained for detection.

    Only the reference-reviewed whole-second grid is scored for detection. A
    missing prediction frame is treated as having no detections (FN applies).
    Predictions on other frames are not FP because those frames lack reference.
    Simultaneous-ID conflicts are cached over all supplied in-window frames.
    """
    root = Path(root)
    dataset = root if (root / "observations.csv").is_file() else root / "dataset-2fps"
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise ValueError("Expected finite start < end")
    if not math.isfinite(max_gap_s) or max_gap_s < 0:
        raise ValueError("max_gap_s must be finite and non-negative")

    def in_scope(timestamp):
        return start <= timestamp < end and (
            exclude_window is None or not exclude_window[0] <= timestamp < exclude_window[1]
        )

    ref_frames = _read_csv(dataset / "frames.csv")
    ref_observations = _read_csv(dataset / "observations.csv")
    excluded_groups, excluded_locals = set(), set()
    review_path = dataset / "links_review.csv"
    if review_path.is_file():
        for row in _read_csv(review_path):
            if row.get("status", "").lower() not in {"unresolved", "uncertain", "candidate"}:
                continue
            for column in ("left_id", "right_id", "local_id", "candidate_global_id"):
                identity = row.get(column, "")
                if identity.startswith("G"):
                    excluded_groups.add(identity)
                elif identity.startswith("C"):
                    excluded_locals.add(identity)

    by_person = defaultdict(list)
    for row in ref_observations:
        if row.get("global_id") and row.get("clock_time"):
            by_person[row["global_id"]].append(row)
    episode_for = {}
    episodes_per_person = {}
    for person, rows in by_person.items():
        rows.sort(key=lambda r: (_clock(r["clock_time"]), int(r["camera_id"]), int(r["frame_index"])))
        episode, previous = 0, None
        for row in rows:
            timestamp = _clock(row["clock_time"])
            if previous is None or timestamp - previous > max_gap_s:
                episode += 1
            episode_for[int(row["camera_id"]), int(row["frame_index"]), row["local_id"]] = f"{person}:E{episode:03d}"
            previous = timestamp
        episodes_per_person[person] = episode

    truth = defaultdict(list)
    for row in ref_observations:
        truth[int(row["camera_id"]), int(row["frame_index"])].append(row)

    predicted = {}
    prediction_cameras = defaultdict(set)
    conflicts = Counter()
    all_prediction_frames_in_window = 0
    for frame in frames:
        camera, frame_index = int(frame["camera"]), int(frame["frame_index"])
        frame_key = camera, frame_index
        if frame_key in predicted:
            raise ValueError(f"Duplicate prediction frame: {frame_key}")
        tracks = []
        for track in frame["tracks"]:
            identity = str(track["id"])
            prediction_cameras[identity].add(camera)
            tracks.append({"id": identity, "box": _box(track["box"])})
        ids = [track["id"] for track in tracks]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate local ID in prediction frame: {frame_key}")
        predicted[frame_key] = tracks
        if in_scope(float(frame["time"])):
            all_prediction_frames_in_window += 1
            for left, right in combinations(sorted(ids), 2):
                conflicts[camera, left, right] += 1
    ambiguous_ids = sorted(identity for identity, cameras in prediction_cameras.items() if len(cameras) > 1)
    if ambiguous_ids:
        raise ValueError(f"Prediction local IDs must be camera-scoped: {ambiguous_ids[:10]}")

    totals = Counter()
    per_camera = defaultdict(Counter)
    matched_counts = Counter()
    reference_counts_all = Counter()
    prediction_counts_all = Counter()
    match_rows = []
    standard_frames = []
    matched_unknown, total_unknown = 0, 0
    visited_ref_frames = set()
    for frame in ref_frames:
        if (frame["review_status"] != "reviewed" or not frame.get("clock_time")
                or int(frame["sample_ms"]) % 1000):
            continue
        timestamp = _clock(frame["clock_time"])
        if not in_scope(timestamp):
            continue
        camera, frame_index = int(frame["camera_id"]), int(frame["frame_index"])
        frame_key = camera, frame_index
        if frame_key in visited_ref_frames:
            raise ValueError(f"Duplicate source frame in reference grid: {frame_key}")
        visited_ref_frames.add(frame_key)
        gt = truth.get(frame_key, [])
        pred = predicted.get(frame_key, [])
        gt_boxes = [_box([r["x_min"], r["y_min"], r["x_max"], r["y_max"]]) for r in gt]
        matches = _match(gt_boxes, [r["box"] for r in pred])
        counts = {"tp": len(matches), "fp": len(pred) - len(matches), "fn": len(gt) - len(matches),
                  "reference_frames": 1, "reference_boxes": len(gt), "prediction_boxes": len(pred),
                  "missing_prediction_frames": int(frame_key not in predicted)}
        totals.update(counts)
        per_camera[camera].update(counts)
        known = [bool(row.get("global_id")) and row["global_id"] not in excluded_groups
                 and row["local_id"] not in excluded_locals for row in gt]
        total_unknown += sum(not value for value in known)
        for row, is_known in zip(gt, known, strict=True):
            if is_known:
                reference_counts_all[episode_for[camera, frame_index, row["local_id"]], camera] += 1
        unknown_matches = {pi for gi, pi, _ in matches if not known[gi]}
        if include_standard_frames:
            gt_indices = [i for i, is_known in enumerate(known) if is_known]
            prediction_indices = [i for i in range(len(pred)) if i not in unknown_matches]
            standard_frames.append((
                [episode_for[camera, frame_index, gt[i]["local_id"]] for i in gt_indices],
                [pred[i]["id"] for i in prediction_indices],
                _iou_matrix([gt_boxes[i] for i in gt_indices], [pred[i]["box"] for i in prediction_indices]),
            ))
        for pi, track in enumerate(pred):
            if pi not in unknown_matches:
                prediction_counts_all[track["id"], camera] += 1
        for gi, pi, iou in matches:
            row, prediction = gt[gi], pred[pi]
            if not known[gi]:
                matched_unknown += 1
                continue
            episode = episode_for[camera, frame_index, row["local_id"]]
            matched_counts[prediction["id"], episode, camera] += 1
            match_rows.append({"prediction_id": prediction["id"], "reference_episode": episode,
                               "reference_person": row["global_id"], "reference_local_id": row["local_id"],
                               "camera": camera, "frame_index": frame_index, "time": timestamp, "iou": iou})

    detection = {**dict(totals), **_prf(totals["tp"], totals["fp"], totals["fn"]),
                 "iou_threshold": IOU_THRESHOLD,
                 "scope": "reviewed whole-second reference grid, exact source indices in [start,end)"}
    return {
        "start": float(start), "end": float(end), "max_gap_s": float(max_gap_s),
        "excluded_window": exclude_window,
        "detection": detection,
        "detection_by_camera": {str(c): {**dict(v), **_prf(v["tp"], v["fp"], v["fn"])}
                                for c, v in sorted(per_camera.items())},
        "matched_counts": [{"prediction_id": p, "reference_episode": g, "camera": c, "n": n}
                           for (p, g, c), n in sorted(matched_counts.items())],
        "reference_counts_all": dict(reference_counts_all),
        "prediction_counts_all": dict(prediction_counts_all),
        "matches": match_rows,
        "standard_frames": standard_frames,
        "simultaneous_local_pairs": [{"camera": c, "left_id": a, "right_id": b, "frame_count": n}
                                     for (c, a, b), n in sorted(conflicts.items())],
        "prediction_frames_in_window": all_prediction_frames_in_window,
        "reference_episodes_per_person_full_recording": episodes_per_person,
        "association_exclusions": {"global_ids": sorted(excluded_groups),
                                   "local_ids": sorted(excluded_locals),
                                   "reference_observations_excluded": total_unknown,
                                   "matched_observations_excluded": matched_unknown},
        "limitations": [
            "Association pair metrics are conditional on matched detections, not IDF1/HOTA.",
            "IoU compares visible-part reference boxes with detector boxes; partial figures may be penalized.",
            "Screen-clock alignment is approximate; epoch representation does not establish real UTC.",
            "Unknown groups/pairs are excluded rather than treated as different people.",
        ],
    }


def measure(data: dict, assignment: dict[str, str]) -> dict:
    """Evaluate an ID assignment without repeating box matching or video work.

    assignment maps camera-scoped local node IDs to proposed global IDs. A
    missing mapping leaves that local node separate; it is never silently
    grouped with another unassigned node. Pair units are unordered pairs of
    matched observations from DIFFERENT cameras, across the selected window.
    """
    def group(local: str) -> tuple[str, str]:
        value = assignment.get(local)
        return ("global", str(value)) if value is not None else ("local", local)

    def label(identity: tuple[str, str]) -> str:
        return identity[1] if identity[0] == "global" else f"UNASSIGNED:{identity[1]}"

    predicted_counts = defaultdict(Counter)
    reference_counts = defaultdict(Counter)
    intersection_counts = defaultdict(Counter)
    refs_by_prediction, predictions_by_ref = defaultdict(set), defaultdict(set)
    count_by_prediction_ref = Counter()
    for row in data["matched_counts"]:
        proposed = group(row["prediction_id"])
        reference, camera, n = row["reference_episode"], int(row["camera"]), int(row["n"])
        predicted_counts[proposed][camera] += n
        reference_counts[reference][camera] += n
        intersection_counts[proposed, reference][camera] += n
        refs_by_prediction[proposed].add(reference)
        predictions_by_ref[reference].add(proposed)
        count_by_prediction_ref[proposed, reference] += n

    positive_pred = sum(_cross_pairs(v) for v in predicted_counts.values())
    positives_by_episode = {episode: _cross_pairs(counts) for episode, counts in reference_counts.items()}
    positive_ref = sum(positives_by_episode.values())
    tp_by_episode = Counter()
    for (_, episode), counts in intersection_counts.items():
        tp_by_episode[episode] += _cross_pairs(counts)
    tp = sum(tp_by_episode.values())
    episode_recalls = {episode: tp_by_episode[episode] / count for episode, count in positives_by_episode.items()
                       if count}
    fp, fn = positive_pred - tp, positive_ref - tp
    if fp < 0 or fn < 0:
        raise AssertionError("Pair count inconsistency")
    reference_all, prediction_all = defaultdict(Counter), defaultdict(Counter)
    for (episode, camera), count in data["reference_counts_all"].items():
        reference_all[episode][camera] += count
    for (local, camera), count in data["prediction_counts_all"].items():
        prediction_all[group(local)][camera] += count
    fp_all = sum(_cross_pairs(v) for v in prediction_all.values()) - tp
    fn_all = sum(_cross_pairs(v) for v in reference_all.values()) - tp
    split = {g: sorted(label(p) for p in ps) for g, ps in predictions_by_ref.items() if len(ps) > 1}
    merged = {label(p): sorted(gs) for p, gs in refs_by_prediction.items() if len(gs) > 1}
    conflict_examples, conflict_frames, conflict_pairs = [], 0, 0
    for row in data["simultaneous_local_pairs"]:
        if group(row["left_id"]) == group(row["right_id"]):
            conflict_frames += int(row["frame_count"])
            conflict_pairs += 1
            if len(conflict_examples) < 20:
                conflict_examples.append({**row, "global_id": label(group(row["left_id"]))})
    matched_n = sum(int(row["n"]) for row in data["matched_counts"])
    pure_n = sum(max(count_by_prediction_ref[p, g] for g in gs) for p, gs in refs_by_prediction.items())
    return {
        "detection": dict(data["detection"]),
        "detection_by_camera": data["detection_by_camera"],
        "association": {
            "scope": "cross_camera_matched_observations_only",
            "tp_pairs": tp, "fp_pairs": fp, "fn_pairs": fn,
            "predicted_positive_pairs": positive_pred, "reference_positive_pairs": positive_ref,
            **_prf(tp, fp, fn),
            "matched_observations": matched_n,
            "reference_episodes": len(reference_counts),
            "reference_episodes_on_multiple_cameras": len(episode_recalls),
            "reference_episodes_with_correct_cross_camera_pair": sum(value > 0 for value in episode_recalls.values()),
            "macro_episode_pair_recall": sum(episode_recalls.values()) / len(episode_recalls) if episode_recalls else None,
            "prediction_groups": len(predicted_counts),
            "max_gap_s": data["max_gap_s"],
        },
        "end_to_end_association": {
            "scope": "cross_camera_observation_pairs_in_reviewed_frames_with_box_errors",
            "tp_pairs": tp, "fp_pairs": fp_all, "fn_pairs": fn_all,
            **_prf(tp, fp_all, fn_all),
        },
        "diagnostics": {
            "split_reference_episodes": len(split),
            "extra_prediction_groups_from_splits": sum(len(v) - 1 for v in split.values()),
            "merged_prediction_groups": len(merged),
            "matched_observation_purity": pure_n / matched_n if matched_n else None,
            "split_examples": dict(list(sorted(split.items()))[:20]),
            "merge_examples": dict(list(sorted(merged.items()))[:20]),
            "same_camera_simultaneous_conflict_frame_pairs": conflict_frames,
            "same_camera_simultaneous_conflicting_local_pairs": conflict_pairs,
            "conflict_scope": "all supplied in-scope prediction frames, including unlabelled frames",
            "conflict_examples": conflict_examples,
        },
        "association_exclusions": data["association_exclusions"],
        "limitations": data["limitations"],
    }


def standard_metrics(frames, assignment):
    gt_ids, tracker_ids, similarities = [], [], []
    gt_lookup, tracker_lookup = {}, {}
    duplicate_gt_frames = duplicate_prediction_frames = 0
    for truth, local, iou in frames:
        proposed = [assignment.get(identity, identity) for identity in local]
        duplicate_gt_frames += len(set(truth)) != len(truth)
        duplicate_prediction_frames += len(set(proposed)) != len(proposed)
        gt_ids.append(np.array([gt_lookup.setdefault(name, len(gt_lookup)) for name in truth], dtype=int))
        tracker_ids.append(np.array([tracker_lookup.setdefault(name, len(tracker_lookup)) for name in proposed], dtype=int))
        similarities.append(iou)
    if duplicate_gt_frames or duplicate_prediction_frames:
        return {'status': 'invalid_duplicate_id_in_camera_frame', 'duplicate_gt_frames': duplicate_gt_frames,
                'duplicate_prediction_frames': duplicate_prediction_frames}
    data = {
        'num_gt_ids': len(gt_lookup), 'num_tracker_ids': len(tracker_lookup),
        'num_gt_dets': sum(map(len, gt_ids)), 'num_tracker_dets': sum(map(len, tracker_ids)),
        'num_timesteps': len(frames), 'gt_ids': gt_ids, 'tracker_ids': tracker_ids,
        'similarity_scores': similarities,
    }
    hota = HOTA().eval_sequence(data)
    identity = Identity({'THRESHOLD': 0.5, 'PRINT_CONFIG': False}).eval_sequence(data)
    return {
        'status': 'ok', 'camera_frames': len(frames),
        **{key: float(np.mean(hota[key])) for key in ('HOTA', 'DetA', 'AssA', 'LocA')},
        **{key: float(identity[key]) for key in ('IDF1', 'IDP', 'IDR')},
        **{key: int(identity[key]) for key in ('IDTP', 'IDFP', 'IDFN')},
    }


def observations_by_group(tracks, assignment, start):
    support = defaultdict(set)
    for local, group in assignment.items():
        support[group].update(int(round((time - start) * BIN_HZ)) for _, _, time, _ in tracks[local])
    return support


def confirmed_breakdown(data, assignment, support):
    """Separate the net difference by components of the prediction-to-reference overlap graph."""
    neighbors = defaultdict(set)
    group_votes = defaultdict(Counter)
    for row in data['matched_counts']:
        group = assignment[row['prediction_id']]
        reference = row['reference_episode']
        neighbors['p', group].add(('r', reference))
        neighbors['r', reference].add(('p', group))
        group_votes[group][reference.split(':', 1)[0]] += row['n']

    reviewed = {assignment[local] for local, _ in data['prediction_counts_all']}
    matched = set(group_votes)
    membership = {
        'confirmed_match': matched, 'reviewed_no_confirmed_match': reviewed - matched,
        'outside_strict_review': set(assignment.values()) - reviewed,
    }
    distribution = {name: {str(bins): sum(len(support[group]) < bins for group in groups)
                           for bins in MIN_BINS if bins}
                    for name, groups in membership.items()}

    remaining = set(neighbors)
    components = defaultdict(Counter)
    while remaining:
        stack = [remaining.pop()]
        size = Counter()
        while stack:
            node = stack.pop()
            size[node[0]] += 1
            new_nodes = neighbors[node] & remaining
            remaining -= new_nodes
            stack.extend(new_nodes)
        p, r = size['p'], size['r']
        kind = ('one_to_one' if p == r == 1 else 'pure_split' if r == 1 else
                'pure_merge' if p == 1 else 'mixed')
        components[kind].update(components=1, predicted=p, reference=r, net=p-r)
    return {
        'components': dict(components),
        'short_ids_by_group_status': distribution,
        'mixed_people_ids': sum(len(votes) > 1 for votes in group_votes.values()),
        'wrong_person_observations': sum(sum(votes.values()) - max(votes.values())
                                         for votes in group_votes.values()),
    }


def evaluate_predictions(predictions, reference, *, start=None, end=None):
    """Evaluate source-frame boxes and global assignments against the Δ60 reference."""
    start = predictions['recording_start'] if start is None else start
    frames = predictions['frames']
    if end is None:
        end = max((frame['time'] for frame in frames), default=start) + 1 / predictions['sampling_hz']
    assignment = {}
    tracks = defaultdict(list)
    for frame in frames:
        if not start <= frame['time'] < end:
            continue
        camera = frame['camera']
        for track in frame['tracks']:
            local = track['id']
            group = track['global_id']
            if assignment.setdefault(local, group) != group:
                raise ValueError(f'Local track {local} has conflicting global IDs')
            tracks[local].append((camera, frame['frame_index'], frame['time'], track['box']))
    data = evaluation_data(reference, frames, start, end, 60, include_standard_frames=True)
    paired = measure(data, assignment)
    support = observations_by_group(tracks, assignment, start)
    return {
        'window': [start, end],
        'global_ids': len(set(assignment.values())),
        'local_tracks': len(assignment),
        'matched_reference_episodes': len({row['reference_episode'] for row in data['matched_counts']}),
        'pair_metrics': paired['end_to_end_association'],
        'tracking': standard_metrics(data['standard_frames'], assignment),
        'detection': data['detection'],
        'diagnosis': confirmed_breakdown(data, assignment, support),
    }
