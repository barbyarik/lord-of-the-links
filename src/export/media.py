"""Prepare seekable browser video while preserving the source video timeline."""
from fractions import Fraction
from pathlib import Path
from shutil import copyfile


def prepare_media(source, directory, camera, offset=None):
    source = Path(source).resolve(strict=True)
    suffix = source.suffix.lower()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'camera-{camera}{suffix if suffix in {".mp4", ".webm"} else ".mp4"}'
    temporary = destination.with_suffix('.tmp' + destination.suffix)
    if source == destination.resolve():
        return destination, 0.0 if offset is None else offset, media_metadata(source)
    try:
        if suffix in {'.mp4', '.webm'}:
            copyfile(source, temporary)
            origin, metadata = (0.0 if offset is None else offset), media_metadata(source)
        else:
            origin, metadata = convert_to_mp4(source, temporary, offset)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination, origin, metadata



def media_metadata(source):
    try:
        import av
    except ImportError:
        return None
    with av.open(str(source)) as video:
        stream = video.streams.video[0]
        if video.duration is None:
            return None
        return stream.width, stream.height, round(video.duration / 1000)


def convert_to_mp4(source, destination, offset):
    try:
        import av
    except ImportError as exc:
        raise RuntimeError('Install the inference requirements to convert native camera video to MP4') from exc
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream), None)
        if first is None or first.time is None:
            raise ValueError(f'No timestamped video frames in {source}')
        origin = first.time
    if offset is not None and abs(offset - origin) > .001:
        raise ValueError(f'Media offset for {source.name} must equal its first decoded PTS ({origin:.3f} s)')

    with av.open(str(source)) as container, av.open(
            str(destination), 'w', options={'movflags': '+faststart', 'video_track_timescale': '1000'}) as output:
        stream = container.streams.video[0]
        if stream.codec_context.name == 'h264':
            remux_h264(container, stream, output, origin)
        else:
            encode_h264(container, stream, output, origin)

    with av.open(str(destination)) as output:
        stream = output.streams.video[0]
        first = next(output.decode(stream), None)
        if first is None or abs(first.time) > .001:
            raise ValueError(f'Failed to normalize the video start for {source.name}')
        return origin, (stream.width, stream.height, round(output.duration / 1000))


def remux_h264(container, stream, output, origin):
    target = output.add_stream_from_template(stream)
    offset = round(origin / stream.time_base)
    for packet in container.demux(stream):
        if packet.pts is None or packet.dts is None or packet.pts < offset:
            continue
        packet.pts -= offset
        packet.dts -= offset
        packet.stream = target
        output.mux(packet)


def encode_h264(container, stream, output, origin):
    target = output.add_stream('libx264', rate=stream.average_rate or 25)
    scale = min(1, 1296 / stream.width)
    target.width = max(2, 2 * round(stream.width * scale / 2))
    target.height = max(2, 2 * round(stream.height * scale / 2))
    target.pix_fmt = 'yuv420p'
    target.codec_context.options = {'preset': 'ultrafast', 'crf': '24'}
    clock = Fraction(1, 1000)
    target.time_base = clock
    target.codec_context.time_base = clock
    stream.thread_type = 'AUTO'
    for frame in container.decode(stream):
        if frame.time is None or frame.time < origin:
            continue
        pts = round((frame.time - origin) / clock)
        frame = frame.reformat(width=target.width, height=target.height, format=target.pix_fmt)
        frame.time_base = clock
        frame.pts = pts
        for packet in target.encode(frame):
            output.mux(packet)
    for packet in target.encode():
        output.mux(packet)
