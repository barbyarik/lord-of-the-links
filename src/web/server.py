"""Serve a generated LoL demo or an existing 2 Hz reference on localhost."""
import argparse
import json
import mimetypes
import shutil
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from .dataset_io import DatasetError, load_dataset
from .ui_dataset import build_ui

STATIC = Path(__file__).resolve().parent / 'static'
FLOOR_FILES = ('tracks-floor.json', 'calibration.json', 'plan.svg')


def dataset_root(folder):
    """Accept a flat export or an archive with one enclosing directory."""
    for name in ('dataset.json', 'frames.csv', 'dataset-2fps/frames.csv'):
        if (folder / name).is_file():
            return folder
    children = [path for path in folder.iterdir() if path.is_dir() and path.name != '__MACOSX']
    if len(children) == 1:
        return dataset_root(children[0])
    raise DatasetError(f'No demo bundle or 2 Hz reference export in {folder}')


def resolve_dataset(source, extraction_dir):
    source = Path(source).resolve(strict=True)
    if source.is_dir():
        return dataset_root(source)
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            target = (extraction_dir / entry.filename).resolve()
            if not target.is_relative_to(extraction_dir.resolve()):
                raise DatasetError(f'Unsafe path in {source.name}')
            if not entry.is_dir():
                archive.extract(entry, extraction_dir)
    return dataset_root(extraction_dir)


def prepare(data, floor=None):
    data = Path(data).resolve()
    if (data / 'dataset.json').is_file():
        payload = json.loads((data / 'dataset.json').read_text(encoding='utf-8'))
        media = {}
        for camera in payload['manifest']['cameras']:
            video = within(data, camera['video'])
            if video is None:
                raise DatasetError(f"Missing video for camera {camera['camera_id']}: {camera['video']}")
            media[f"/data/{camera['video']}"] = video
    else:
        payload = build_ui(load_dataset(data))
        media = {url: Path(file) for url, file in payload.pop('_media_files').items()}
        for camera in payload['manifest']['cameras']:
            video = camera['video']
            if not video or video.startswith('/'):
                continue
            file = within(data, video)
            if file is None:
                raise DatasetError(f"Missing video for camera {camera['camera_id']}: {video}")
            media[f'/data/{video}'] = file
    floor = Path(floor) if floor is not None else data / 'floor'
    files = {f'/floor/{name}': floor / name for name in FLOOR_FILES if (floor / name).is_file()}
    generated = {}
    if len(files) != len(FLOOR_FILES):
        if payload['manifest']['source_format'] == 'predictions' or floor != data / 'floor':
            raise DatasetError(f'Missing floor handoff files in {floor}')
        payload['manifest']['spatial_available'] = False
        generated['/floor/tracks-floor.json'] = ('application/json', b'{"tracks":{}}')
        generated['/floor/calibration.json'] = ('application/json', b'{"anchors":{"walkable":[]}}')
        generated['/floor/plan.svg'] = ('image/svg+xml', b'<svg xmlns="http://www.w3.org/2000/svg"/>')
    generated['/dataset.json'] = json_response(payload)
    generated['/validation.json'] = json_response(payload['validation'])
    return payload, {**media, **files}, generated


def json_response(value):
    return 'application/json; charset=utf-8', json.dumps(
        value, ensure_ascii=False, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')


def within(root, relative):
    target = (root / relative).resolve()
    return target if target.is_relative_to(root) and target.is_file() else None


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, files, generated, **kwargs):
        self.files, self.generated = files, generated
        super().__init__(*args, **kwargs)

    def send_head(self):
        url = unquote(urlsplit(self.path).path)
        self.remaining = None
        if url in self.generated:
            mime, body = self.generated[url]
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            return BytesIO(body)

        file = self.files.get(url) or within(STATIC, url.lstrip('/') or 'index.html')
        if file is None:
            self.send_error(404)
            return None
        size = file.stat().st_size
        if self.headers.get('Range'):
            try:
                start, end = parse_byte_range(self.headers['Range'], size)
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return None
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.remaining = end - start + 1
        else:
            start = 0
            self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(size if self.remaining is None else self.remaining))
        self.send_header('Accept-Ranges', 'bytes')
        self.end_headers()
        stream = file.open('rb')
        stream.seek(start)
        return stream

    def copyfile(self, source, output):
        try:
            if self.remaining is None:
                shutil.copyfileobj(source, output)
            else:
                while self.remaining:
                    chunk = source.read(min(self.remaining, 1024 * 1024))
                    if not chunk:
                        break
                    output.write(chunk)
                    self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()


def parse_byte_range(value, size):
    unit, spec = value.split('=', 1)
    if unit != 'bytes' or ',' in spec:
        raise ValueError('Only one byte range is supported')
    left, right = spec.split('-')
    if not left and (not right or int(right) <= 0):
        raise ValueError('Empty suffix range')
    start = int(left) if left else max(0, size - int(right))
    end = min(int(right), size - 1) if left and right else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError('Unsatisfiable byte range')
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Demo directory or 2 Hz reference directory/ZIP')
    parser.add_argument('--floor', type=Path, help='Optional map handoff for a standalone 2 Hz reference')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    with TemporaryDirectory(prefix='lol-reference-') as temp:
        try:
            data = resolve_dataset(args.dataset, Path(temp))
            payload, files, generated = prepare(data, args.floor)
        except (DatasetError, OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            parser.error(str(exc))
        print(f"Loaded {len(payload['observations'])} observations, {len(payload['tracks'])} IDs", flush=True)
        if args.prepare_only:
            return
        handler = partial(DemoHandler, files=files, generated=generated)
        with ThreadingHTTPServer((args.host, args.port), handler) as server:
            print(f'The Lord of the Links → http://{args.host}:{server.server_port}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
