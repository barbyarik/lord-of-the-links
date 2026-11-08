"""Display-time mapping for repeated camera-clock readings."""


def disambiguate_clock(frames, camera_ids):
    """Adjust only playback positions on repeated clock labels, preserving field 5."""
    adjusted = 0
    for camera_id in camera_ids:
        rows = sorted(
            (row for row in frames if row[0] == camera_id and row[1] is not None and row[2] is not None),
            key=lambda row: row[2],
        )
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end][1] == rows[start][1]:
                end += 1
            for index in range(start + 1, end):
                offset = rows[index][2] - rows[start][2]
                if end < len(rows):
                    span = rows[end][2] - rows[start][2]
                    offset *= (rows[end][1] - rows[start][1]) / span
                rows[index][1] = rows[start][1] + offset
                rows[index][12] = 'ui_disambiguated_clock'
                adjusted += 1
            start = end
    return adjusted
