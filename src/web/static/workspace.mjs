// Layout is independent of tracking and analytics. Each drawer has one state owner.
export function mapInsets(width, drawers, cameraWidth) {
  const compact = width <= 760;
  let left = 22;
  if (compact) left = 8;
  else if (drawers.results) left = width <= 1150 ? 320 : 360;
  else if (drawers.tracks) left = 255;

  const right = compact ? 8 : drawers.cameras
    ? Math.min(cameraWidth + 48, Math.max(22, width - left - 180))
    : 22;
  const bottom = drawers.presence ? Math.round(Math.min(430, drawers.height * .43) + 162) : 145;
  return {left, right, bottom};
}

export function createWorkspace(onChange) {
  const $ = selector => document.querySelector(selector);
  const cameraPanel = $('.camera-panel');
  const layoutSelect = $('#cameraLayout');
  const state = {
    tracks: false, cameras: true, results: false, presence: false,
    height: innerHeight, layer: 'where', cameraLayout: 'compact', lastSplitLayout: 'compact',
  };
  const drawerButtons = {
    tracks: $('#tracksToggle'),
    results: $('#analysisToggle'),
    presence: $('#presenceToggle'),
    cameras: $('#cameraDrawerToggle'),
  };

  function apply() {
    state.height = innerHeight;
    const full = state.cameraLayout === 'full';
    document.body.dataset.cameraLayout = state.cameraLayout;
    layoutSelect.value = state.cameraLayout;
    $('#tracksDrawer').inert = !state.tracks;
    cameraPanel.inert = !state.cameras;
    document.body.classList.toggle('tracks-open', state.tracks);
    document.body.classList.toggle('camera-open', state.cameras);
    document.body.classList.toggle('results-open', state.results && !full);

    const inset = mapInsets(innerWidth, state, cameraPanel.getBoundingClientRect().width);
    for (const side of ['left', 'right', 'bottom']) {
      document.documentElement.style.setProperty('--map-' + side, inset[side] + 'px');
    }
    $('#analysis').hidden = full || !state.results || state.layer === 'where';
    $('#presence').hidden = !state.presence;
    for (const [name, button] of Object.entries(drawerButtons)) {
      button.setAttribute('aria-expanded', String(state[name]));
    }
    drawerButtons.cameras.querySelector('b').textContent = state.cameras ? '›' : '‹';
    drawerButtons.results.hidden = full || state.layer === 'where';
    drawerButtons.results.textContent = state.layer === 'why' ? 'Пики' : 'Фрагменты';
    onChange();
  }

  function set(name, value) {
    state[name] = value;
    if (value && name === 'tracks') state.results = false;
    if (value && name === 'results') state.tracks = false;
    if (innerWidth < 1000 && value && name === 'results') state.cameras = false;
    if (innerWidth < 1000 && value && name === 'cameras') state.results = false;
    apply();
  }

  for (const [name, button] of Object.entries(drawerButtons)) {
    button.onclick = () => set(name, !state[name]);
  }
  layoutSelect.onchange = () => {
    state.cameraLayout = layoutSelect.value;
    if (state.cameraLayout !== 'full') state.lastSplitLayout = state.cameraLayout;
    apply();
  };
  $('#tracksClose').onclick = () => set('tracks', false);
  $('#presenceClose').onclick = () => set('presence', false);
  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    state.tracks = false;
    state.results = false;
    state.presence = false;
    if (state.cameraLayout === 'full') state.cameraLayout = state.lastSplitLayout;
    apply();
  });
  window.addEventListener('resize', apply);

  return {
    state,
    set,
    apply,
    activateLayer(layer) {
      state.layer = layer;
      state.results = layer !== 'where';
      state.cameras = layer === 'where';
      state.tracks = false;
      state.presence = false;
      state.cameraLayout = state.lastSplitLayout;
      apply();
    },
  };
}
