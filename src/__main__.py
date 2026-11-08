"""Run inference, score predictions or build an offline viewer bundle."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SITE = ROOT / 'config/site/hackathon.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def cameras(values):
    paths = {}
    for value in values or ():
        number, delimiter, file = value.partition('=')
        if not delimiter or not number.isdecimal() or int(number) <= 0 or not file:
            raise ValueError(f'Expected CAMERA=PATH, got {value!r}')
        camera = int(number)
        if camera in paths:
            raise ValueError(f'Duplicate video for camera {camera}')
        paths[camera] = Path(file)
    return paths


def run(args):
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.local/yolo'))
    from .pipeline import run_pipeline

    site = read_json(args.site)
    if args.recording:
        recording = read_json(args.recording)
        site |= {key: recording[key] for key in ('recording_start', 'recording_end', 'cameras')
                 if key in recording}
    if args.scene_cache:
        site['scene'] = read_json(args.scene_cache)
    for path in (args.output, args.save_detections, args.save_appearance):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    result = run_pipeline(
        cameras(args.video), site=site, profile=args.profile, model_dir=args.models,
        detection_cache=args.detections, appearance_cache=args.appearance,
        output=args.output, device=args.device, detection_cache_output=args.save_detections,
        appearance_cache_output=args.save_appearance, progress=lambda line: print(line, flush=True),
    )
    ids = {track['global_id'] for frame in result['frames'] for track in frame['tracks']}
    print(f'{len(result["frames"])} sampled frames, {len(ids)} global IDs → {args.output}')


def evaluate(args):
    from .evaluation import evaluate_predictions

    result = evaluate_predictions(read_json(args.predictions), args.reference, start=args.start, end=args.end)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')


def build_demo(args):
    from .export.bundle import build_bundle

    output = build_bundle(args.predictions, args.output, calibration=args.calibration,
                          plan=args.plan, media=cameras(args.video), smoothing_sigma=args.sigma)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    inference = commands.add_parser('run', help='Run a preset from videos or explicit caches')
    inference.set_defaults(handler=run)
    inference.add_argument('--site', type=Path, default=DEFAULT_SITE)
    inference.add_argument('--recording', type=Path)
    inference.add_argument('--profile', default='balanced', help='Preset name or path to a profile JSON')
    inference.add_argument('--video', action='append', metavar='CAMERA=PATH')
    inference.add_argument('--models', type=Path)
    inference.add_argument('--device', default='cpu', help='cpu, mps or a CUDA device such as cuda:0')
    inference.add_argument('--detections', type=Path)
    inference.add_argument('--appearance', type=Path)
    inference.add_argument('--scene-cache', type=Path)
    inference.add_argument('--save-detections', type=Path)
    inference.add_argument('--save-appearance', type=Path)
    inference.add_argument('--output', required=True, type=Path)

    scoring = commands.add_parser('evaluate', help='Score a run on the reviewed reference grid')
    scoring.set_defaults(handler=evaluate)
    scoring.add_argument('--predictions', required=True, type=Path)
    scoring.add_argument('--reference', required=True, type=Path)
    scoring.add_argument('--start', type=float)
    scoring.add_argument('--end', type=float)
    scoring.add_argument('--output', type=Path)

    demo = commands.add_parser('build-demo', help='Export an offline bundle from predictions and source media')
    demo.set_defaults(handler=build_demo)
    demo.add_argument('--predictions', required=True, type=Path)
    demo.add_argument('--output', required=True, type=Path)
    demo.add_argument('--calibration', type=Path, default=ROOT / 'config/site/calibration.json')
    demo.add_argument('--plan', type=Path)
    demo.add_argument('--video', action='append', required=True, metavar='CAMERA=PATH')
    demo.add_argument('--sigma', type=float, default=.32, help='Map display smoothing in seconds')
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
