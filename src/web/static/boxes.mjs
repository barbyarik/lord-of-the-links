import {color, escape as escapeHtml} from './map.mjs';

// Both native video and reduced JPEG previews use source-resolution coordinates.
export function boxesMarkup(rows, selectedId) {
  return rows.map(row => {
    const [,,,, x1, y1, x2, y2, status] = row;
    const id = row[3];
    const label = id + (status === 'uncertain' ? ' ?' : '');
    const opacity = !selectedId || selectedId === id ? 1 : .24;
    const stroke = color(id);
    return `<g opacity="${opacity}"><rect x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" fill="none" stroke="${stroke}" stroke-width="13"/><rect x="${x1}" y="${Math.max(0,y1-70)}" width="${label.length*34+20}" height="70" fill="${stroke}"/><text x="${x1+8}" y="${Math.max(52,y1-15)}" font-size="54" fill="#fff">${escapeHtml(label)}</text></g>`;
  }).join('');
}
