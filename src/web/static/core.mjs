export const hasObservations = frame => frame?.[3] === 'reviewed' || frame?.[3] === 'predicted';

export function nearest(rows, time, column = 1) {
  if (!rows.length) return null;
  const rightIndex = lowerBound(rows, time, column);
  const left = rows[Math.max(0, rightIndex - 1)];
  const right = rows[Math.min(rightIndex, rows.length - 1)];
  return Math.abs(left[column] - time) <= Math.abs(right[column] - time) ? left : right;
}

function lowerBound(rows, time, column) {
  let low = 0, high = rows.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (rows[middle][column] < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

export function makeModel(dataset) {
  const frames = {}, videoFrames = {}, clockFrames = {}, obs = new Map();
  const cameras = dataset.manifest?.cameras?.map(camera => camera.camera_id) ?? [1, 2, 3, 4, 5];
  const sampleHz = dataset.manifest?.sampling_hz ?? 2;
  const tolerance = .56 / sampleHz;
  for (const camera of cameras) {
    frames[camera] = dataset.frames
      .filter(frame => frame[0] === camera && Number.isFinite(frame[1]))
      .sort((a, b) => a[1] - b[1]);
    videoFrames[camera] = frames[camera]
      .filter(frame => Number.isFinite(frame[2]) && frame[3] !== 'missing')
      .sort((a, b) => a[2] - b[2]);
    clockFrames[camera] = [...videoFrames[camera]].sort((a, b) => a[1] - b[1]);
  }
  for (const observation of dataset.observations) {
    const key = observation[0] + ':' + observation[9];
    if (!obs.has(key)) obs.set(key, []);
    obs.get(key).push(observation);
  }

  function frameAt(camera, time) {
    const frame = nearest(frames[camera], time);
    return frame && Math.abs(frame[1] - time) <= tolerance ? frame : null;
  }

  function coverage(time) {
    const result = {reviewed: 0, predicted: 0, missing: 0, unreviewed: 0, unmapped: 0};
    for (const camera of cameras) {
      const frame = frameAt(camera, time);
      result[frame ? frame[3] : 'unmapped']++;
    }
    return result;
  }

  function snapshot(time) {
    const result = [];
    for (const camera of cameras) {
      const frame = frameAt(camera, time);
      if (hasObservations(frame)) result.push(...(obs.get(camera + ':' + frame[4]) || []));
    }
    return result;
  }

  function videoTime(camera, time, hold = false) {
    const rows = clockFrames[camera];
    if (!rows.length) return null;
    if (time < rows[0][1] - tolerance) return hold ? rows[0][2] : null;
    if (time > rows.at(-1)[1] + tolerance) return hold ? rows.at(-1)[2] : null;
    const index = lowerBound(rows, time, 1);
    const left = rows[Math.max(0, index - 1)];
    const right = rows[Math.min(index, rows.length - 1)];
    if (right[2] < left[2]) {
      // A clock reversal is not a valid segment for video interpolation.
      const frame = nearest(rows, time);
      return Math.abs(frame[1] - time) <= tolerance ? frame[2] : null;
    }
    const offset = right[1] === left[1] ? time - left[1]
      : (right[2] - left[2]) * (time - left[1]) / (right[1] - left[1]);
    return Math.max(0, left[2] + offset);
  }

  const end = dataset.frames.reduce((max, frame) => Number.isFinite(frame[1]) ? Math.max(max, frame[1]) : max, 0);
  return {frames, videoFrames, frameAt, coverage, obs, snapshot, videoTime, end, sampleHz, tolerance};
}
