import {boxesMarkup} from './boxes.mjs';
import {createWorkspace} from './workspace.mjs';
import {visibilityIntervals,rowLayout,smoothActivity} from './presentation.mjs';
import {hasObservations,makeModel,nearest} from './core.mjs';
import {cleanTracks,currentPoints,frequentFragments,densityGrid,densitySummary} from './spatial.mjs';
import {OfficeMap,color,escape as esc} from './map.mjs';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const elapsed=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const clock=t=>{if(!D?.manifest.common_t0)return 'Часы неизвестны';let parts=(D?.manifest.common_t0||'1970-01-01T00:00:00').slice(11).split(':').map(Number),base=parts[0]*3600+parts[1]*60+parts[2];let s=Math.floor(base+t);return [Math.floor(s/3600)%24,Math.floor(s/60)%60,s%60].map(v=>String(v).padStart(2,'0')).join(':');};
const statusNames={prediction:'Предикт модели',linked_visual:'Подтверждён',uncertain:'Спорная связь',local_only:'Локальный трек'};
const state={layer:'where',time:180,playing:false,speed:1,camera:0,track:null,allPaths:false,trail:30,window:5,radius:22,heatMode:'dynamic',length:3,fragment:null};
let D,M,raw,tracks,status,map,displayIntervals,fragments=[],peaks=[],heatAll=[],cells=[],last=0,lastRender=0,heatMax=1;
const workspace=createWorkspace(()=>{if(map){syncVideos();renderMap();renderAnalysis();if(!$('#presence').hidden)renderGantt();}});
function fail(message){$('#error').hidden=false;$('#error').textContent=message;}
function rebuildHeat(){let summary=densitySummary(tracks,M.end,state.window,state.radius,M.sampleHz);peaks=summary.peaks;heatAll=summary.cells;heatMax=Math.max(1,...heatAll.map(p=>p.v));}
function rebuildSpatial(){tracks=cleanTracks(raw,false);fragments=frequentFragments(tracks,status,state.length);rebuildHeat();}
function current(){return currentPoints(tracks,state.time,state.camera,M.sampleHz);}
function goto(t){state.time=Math.max(0,Math.min(M.end,t));syncVideos(true);renderTime();renderMap();renderTracks();renderGantt();renderBoxes();}
function choose(id,jump=false){if(!D.tracks.some(t=>t.track_id===id))return;state.track=id;map.zoom=1;if(jump&&state.camera){state.camera=0;renderVideos();}$('#trackSelect').value=id;if(jump&&D.tracks.find(t=>t.track_id===id).time_available!==false)goto(+D.tracks.find(t=>t.track_id===id).first_seen_s);else{renderMap();renderTracks();renderGantt();}renderBoxes();if(state.layer==='where')renderControls();}
function chooseCamera(c){state.camera=c;workspace.set('cameras',true);renderVideos();renderMap();}
function title(){let values={where:['Люди в пространстве','Траектории на плане'],why:['Где собираются люди','Тепловая карта'],how:['Общие пути','Повторяющиеся фрагменты треков']};$('#title').textContent=values[state.layer][0];$('#mapTitle').textContent=values[state.layer][1];$$('[data-layer]').forEach(b=>b.classList.toggle('active',b.dataset.layer===state.layer));}
function layer(l){state.layer=l;workspace.activateLayer(l);title();renderControls();renderAnalysis();renderMap();}
function renderControls(){let e=$('#controls');if(state.layer==='where')e.innerHTML=`<select id="pathMode" aria-label="Период траекторий"><option value="recent">След за последние 30 с</option><option value="all">Траектории за всё время</option></select>${state.track?'<button id="clearTrack">Все треки</button>':''}`;
 if(state.layer==='why')e.innerHTML=`<select id="heatMode" aria-label="Режим тепловой карты"><option value="dynamic">Динамика</option><option value="top">Пики за запись</option></select><select id="window" aria-label="Временное окно"><option value=".5">Текущий момент</option><option value="5">Окно 5 с</option><option value="15">Окно 15 с</option></select><label>Радиус <select id="radius"><option value="15">15</option><option value="22">22</option><option value="32">32</option></select></label>`;
 if(state.layer==='how')e.innerHTML=`<label>Длина фрагмента <select id="length"><option value="2">Короткий</option><option value="3">Средний</option><option value="4">Длинный</option><option value="5">Очень длинный</option></select></label><span class="pill">${fragments.length} общих фрагментов</span>`;
 if($('#pathMode')){$('#pathMode').value=state.allPaths?'all':'recent';$('#pathMode').onchange=e=>{state.allPaths=e.target.value==='all';renderMap();};}
 if($('#clearTrack'))$('#clearTrack').onclick=()=>{state.track=null;$('#trackSelect').value='all';renderGantt();renderControls();renderMap();renderTracks();renderBoxes();};
 if($('#heatMode')){$('#heatMode').value=state.heatMode;$('#window').value=state.window;$('#radius').value=state.radius;$('#heatMode').onchange=e=>{state.heatMode=e.target.value;renderMap();renderAnalysis();};$('#window').onchange=e=>{state.window=+e.target.value;rebuildHeat();renderMap();renderAnalysis();};$('#radius').onchange=e=>{state.radius=+e.target.value;rebuildHeat();renderMap();renderAnalysis();};}
 if($('#length')){$('#length').value=state.length;$('#length').onchange=e=>{state.length=+e.target.value;fragments=frequentFragments(tracks,status,state.length);state.fragment=null;renderControls();renderMap();renderAnalysis();};}
}
function renderMap(){if(!map)return;cells=state.layer==='why'?densityGrid(tracks,state.time,state.window,8,state.radius,M.sampleHz):[];
 if(state.layer==='why'&&state.heatMode==='top')cells=heatAll;
 map.draw({tracks,current:current(),time:state.time,track:state.track,camera:state.camera,showCameras:D.manifest.spatial_available!==false,layer:state.layer,allPaths:state.allPaths,trail:state.trail,cells,fragments,fragment:state.fragment,fieldMax:heatMax});
 $$('#map [data-camera]').forEach(g=>g.onclick=()=>chooseCamera(+g.dataset.camera));$$('#map [data-track]').forEach(g=>g.onclick=()=>choose(g.dataset.track));$$('#map [data-fragment]').forEach(g=>{g.onclick=()=>{state.fragment=g.dataset.fragment;map.zoom=1;renderMap();renderAnalysis();};g.onpointermove=e=>{let f=fragments.find(f=>f.key===g.dataset.fragment);tip(e,`<b>${f.ids.length} эпизодов · ${f.count} проходов</b><small>${f.examples.slice(0,3).map(x=>`${esc(x.id)} · ${clock(x.start)}–${clock(x.end)} · камеры ${x.cameras.join(', ')}`).join('<br>')}</small>`);};g.onpointerleave=hideTip;});
 $('#mapStatus').textContent=D.manifest.spatial_available===false?'Координаты пола не включены в контракт':state.layer==='where'?`${current().length} позиций сейчас${state.track?' · '+state.track:''}`:state.layer==='why'?'Сглаженная концентрация · относительная шкала':`${fragments.length} повторяющихся фрагментов`;
}
function tip(e,html){let r=$('#mapStage').getBoundingClientRect(),t=$('#tip');t.innerHTML=html;t.hidden=false;t.style.left=Math.min(Math.max(8,e.clientX-r.left+12),r.width-310)+'px';t.style.top=Math.min(Math.max(8,e.clientY-r.top+12),r.height-150)+'px';}
function hideTip(){$('#tip').hidden=true;}
function mapPoint(event) {
 const point = new DOMPoint(event.clientX, event.clientY).matrixTransform($('#map').getScreenCTM().inverse());
 return map.planPoint(point.x, point.y);
}
function renderTracks(){let visible=new Set(M.snapshot(state.time).map(o=>o[3])),q=$('#search').value.toLowerCase(),list=D.tracks.filter(t=>t.track_id.toLowerCase().includes(q)&&(!$('#visibleOnly').checked||visible.has(t.track_id)));$('#trackList').innerHTML=list.map(t=>`<button data-id="${esc(t.track_id)}" class="track ${t.track_id===state.track?'active':''}"><span class="dot" style="--c:${color(t.track_id)}"></span><span>${esc(t.track_id)}${t.link_status==='uncertain'?' ?':''}</span><small>${visible.has(t.track_id)?'●':t.time_available===false?'время неизвестно':elapsed(+t.first_seen_s)}</small></button>`).join('')||'<div class="empty">Ничего не найдено</div>';$$('#trackList button').forEach(b=>b.onclick=()=>{choose(b.dataset.id,true);if(state.layer==='where')renderControls();});}
function renderVideos(){let cs=D.manifest.cameras.filter(c=>!state.camera||state.camera===c.camera_id);$('#allCameras').hidden=!state.camera;$('#cameraTitle').textContent=state.camera?'Камера '+state.camera:'Все камеры';$('#cameraTabs').innerHTML=D.manifest.cameras.map(c=>`<button data-camera-tab="${c.camera_id}" class="${state.camera===c.camera_id?'active':''}">${c.camera_id} · ${esc(map.cameras.find(view=>view.id===c.camera_id)?.name ?? c.plan_label)}</button>`).join('');$$('[data-camera-tab]').forEach(b=>b.onclick=()=>chooseCamera(+b.dataset.cameraTab));$('#videos').classList.toggle('single',cs.length===1);$('#videos').innerHTML=cs.map(c=>`<article class="video-card"><div class="video-title"><span>Камера ${c.camera_id}</span><small>Длительность ${elapsed(c.video_duration_ms/1000)}</small></div><div class="video-body">${c.display_mode==='images'?`<img id="preview${c.camera_id}" class="preview-image" alt="Камера ${c.camera_id}: размеченный кадр">`:`<video id="v${c.camera_id}" muted playsinline preload="auto" src="${c.video?.startsWith('/')?c.video:'/data/'+c.video}"></video>`}<svg id="b${c.camera_id}" viewBox="0 0 ${c.bbox_width} ${c.bbox_height}"></svg><div class="video-message" id="msg${c.camera_id}" hidden></div></div></article>`).join('');
 for(let c of cs){let v=$('#v'+c.camera_id);if(!v)continue;v.onloadedmetadata=()=>syncVideos(true);v.onerror=()=>{let msg=$('#msg'+c.camera_id);msg.hidden=false;msg.textContent='Видео не найдено: '+c.video;};}syncVideos(true);}
function syncVideos(force = false) {
 for (const camera of D.manifest.cameras) {
  const video = $('#v' + camera.camera_id);
  if (!video || video.error) continue;
  if (!workspace.state.cameras) { video.pause(); continue; }
  const target = M.videoTime(camera.camera_id, state.time);
  const message = $('#msg' + camera.camera_id);
  const predictions = D.manifest.source_format === 'predictions';
  message.hidden = predictions || target !== null;
  if (!message.hidden) message.textContent = 'Вне доступной записи';
  const position = target ?? (predictions ? M.videoTime(camera.camera_id, state.time, true) : null);
  if (video.readyState && position !== null) {
   const seek = Math.max(0, Math.min(position, video.duration - .03));
   const tolerance = force || target === null ? .01 : Math.max(.14, M.tolerance);
   if (Math.abs(video.currentTime - seek) > tolerance) video.currentTime = seek;
  }
  video.playbackRate = state.speed;
  const play = state.playing && target !== null;
  if (play && video.paused) video.play().catch(() => {});
  if (!play && !video.paused) video.pause();
 }
}
function renderBoxes() {
 for (const camera of D.manifest.cameras) {
  const video = $('#v' + camera.camera_id);
  const boxes = $('#b' + camera.camera_id);
  const preview = $('#preview' + camera.camera_id);
  if (preview) { renderPreview(camera, preview, boxes); continue; }
  if (!video || !boxes) continue;
  const target = M.videoTime(camera.camera_id, state.time);
  const aligned = target !== null && video.readyState >= 2 && !video.seeking &&
    Math.abs(video.currentTime - target) <= Math.max(.16, M.tolerance);
  const frame = nearest(M.videoFrames[camera.camera_id], video.currentTime, 2);
  const valid = hasObservations(frame) && hasObservations(M.frameAt(camera.camera_id, state.time)) &&
    Math.abs(frame[2] - video.currentTime) <= M.tolerance;
  const rows = aligned && valid ? M.obs.get(camera.camera_id + ':' + frame[4]) ?? [] : [];
  boxes.innerHTML = boxesMarkup(rows, state.track);
 }
}
function renderPreview(c,img,b){
 const frame=M.frameAt(c.camera_id,state.time),msg=$('#msg'+c.camera_id);
 if(!frame||!frame[10]||frame[3]==='missing'){
  img.removeAttribute('src');b.innerHTML='';msg.hidden=false;
  msg.textContent=frame?.[3]==='missing'?'Кадр отсутствует':'Нет привязки кадра ко времени';return;
 }
 img.title=`${frame[11]||'Часы не читаются'} · ${frame[6]} · ${frame[12]}`;
 img.onerror=()=>{b.innerHTML='';msg.hidden=false;msg.textContent='Не удалось загрузить превью';};
 if(img.getAttribute('src')!==frame[10]){b.innerHTML='';img.src=frame[10];}
 const ready=img.complete&&img.naturalWidth>0;
 msg.hidden=ready&&frame[3]==='reviewed';
 msg.textContent=img.complete&&!img.naturalWidth?'Не удалось загрузить превью':frame[3]==='unreviewed'?'Кадр не проверен':'Загрузка кадра';
 const rows=ready&&frame[3]==='reviewed'?(M.obs.get(c.camera_id+':'+frame[4])||[]):[];
 b.innerHTML=boxesMarkup(rows,state.track);
}

function renderGantt(){
 const tr=D.tracks.find(t=>t.track_id===state.track),all=!tr;
 const rows=displayIntervals.filter(i=>all||i.track_id===tr.track_id);if(!rows.length){$('#presenceCaption').textContent=tr?tr.track_id:'Все ID';$('#gantt').innerHTML='<div class="empty">Для этого ID нет привязки к общей шкале времени. Наблюдения сохранены в исходном экспорте.</div>';return;}
 const lo=all?0:Math.max(0,Math.min(...rows.map(i=>+i.first_seen_s))-8),hi=all?M.end:Math.min(M.end,Math.max(...rows.map(i=>+i.last_seen_s))+8),x=t=>115+(t-lo)/Math.max(.5,hi-lo)*960;
 let s='',y=12;
 for(const {camera_id:c} of D.manifest.cameras){
  let items=rows.filter(i=>+i.camera_id===c).sort((a,b)=>+a.first_seen_s-+b.first_seen_s),ends=[],placed=[];
  for(let i of items){let lane=0;while(lane<ends.length&&ends[lane]>=+i.first_seen_s)lane++;ends[lane]=+i.last_seen_s;placed.push({i,lane});}
  let layout=rowLayout(ends.length),h=layout.height;
  s+=`<rect x="110" y="${y}" width="980" height="${h}" fill="${c%2?'#142033':'#101925'}"/><text x="0" y="${y+h/2+5}" fill="#91a6c4" font-size="15">Камера ${c}</text>`;
  for(let{i,lane}of placed){let left=x(+i.first_seen_s),w=Math.max(3,x(+i.last_seen_s)-left),top=y+layout.offset+lane*layout.stride;s+=`<g class="segment" data-jump="${i.first_seen_s}" data-episode="${esc(i.track_id)}">${i.observation_count===1?`<circle cx="${left}" cy="${top+10.5}" r="3.5" fill="${color(i.track_id)}"/>`:`<rect x="${left}" y="${top}" width="${w}" height="21" rx="4" fill="${color(i.track_id)}"/>`}<title>${esc(i.track_id)} · камера ${i.camera_id} · ${clock(+i.first_seen_s)}–${clock(+i.last_seen_s)} · ${i.observation_count} кадров</title>${w>58?`<text x="${left+w/2}" y="${top+10.5}" dominant-baseline="central" text-anchor="middle" font-size="12" fill="white">${esc(i.track_id)}</text>`:''}</g>`;}
  y+=h+7;
 }
 if(state.time>=lo&&state.time<=hi)s+=`<line x1="${x(state.time)}" x2="${x(state.time)}" y1="0" y2="${y}" stroke="#0f172a" stroke-width="2"/>`;
 for(let i=0;i<6;i++){let t=lo+(hi-lo)*i/5;s+=`<text x="${x(t)}" y="${y+22}" text-anchor="${i===5?'end':'start'}" font-size="12" fill="#91a6c4">${clock(t)}</text>`;}
 $('#gantt').innerHTML=`<svg style="height:${Math.max(235,y+30)}px" viewBox="0 0 1100 ${y+30}" role="img" aria-label="Интервалы присутствия ID по камерам">${s}</svg>`;
 $('#presenceCaption').innerHTML=all?'Все ID · '+D.tracks.length+' самостоятельных треков':`<span style="color:${color(tr.track_id)}">● ${esc(tr.track_id)}</span> · ${statusNames[tr.link_status]}`;
 $$('#gantt [data-jump]').forEach(g=>{g.onclick=()=>{state.track=g.dataset.episode;$('#trackSelect').value=state.track;goto(+g.dataset.jump);if(state.layer==='where')renderControls();};g.onpointermove=e=>{g.querySelector('rect,circle').setAttribute('stroke','#2563eb');g.querySelector('rect,circle').setAttribute('stroke-width','2');};g.onpointerleave=()=>g.querySelector('rect,circle').removeAttribute('stroke');});
}

function miniMap(points){let xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),loX=Math.min(...xs)-10,loY=Math.min(...ys)-10,w=Math.max(...xs)-loX+10,h=Math.max(...ys)-loY+10;return `<svg viewBox="${loX} ${loY} ${w} ${h}"><polyline points="${points.map(p=>p.join(',')).join(' ')}" fill="none" stroke="#62a2ff" stroke-width="3" stroke-linecap="round"/><circle cx="${points[0][0]}" cy="${points[0][1]}" r="3" fill="#f4f8ff"/></svg>`;}
function renderAnalysis(){let e=$('#analysis');e.hidden=state.layer==='where'||!workspace.state.results||workspace.state.cameraLayout==='full';if(e.hidden)return;if(D.manifest.spatial_available===false){e.innerHTML='<div class="empty">В контракте нет координат пола. Доступны исходные боксы и интервалы ID; для WHY/HOW нужен отдельный согласованный пространственный экспорт.</div>';return;}
 if(state.layer==='why'){e.innerHTML=`<h2>Пики концентрации</h2><div class="cards">${peaks.slice(0,8).map((p,i)=>`<button class="card" data-peak="${i}"><b>${clock(p.time)}</b><small>${esc(map.contextAt(p.x,p.y))}</small><small>Индекс ${p.v.toFixed(1)} · окно ${state.window} с</small></button>`).join('')||'<div class="empty">Нет пространственных наблюдений</div>'}</div>`;$$('[data-peak]').forEach(b=>b.onclick=()=>{state.heatMode='dynamic';goto(peaks[+b.dataset.peak].time);renderControls();});return;}
 let chosen=fragments.find(f=>f.key===state.fragment);e.innerHTML=chosen?`<div class="panel-head"><b>${chosen.ids.length} эпизодов · ${chosen.count} проходов</b><button id="backFragments">Все фрагменты</button></div><div class="cards">${chosen.examples.map((x,i)=>`<button class="card" data-example="${i}"><b>${x.id}</b><small>${clock(x.start)} — ${clock(x.end)}</small><small>Камеры ${x.cameras.join(', ')}</small></button>`).join('')}</div>`:`<h2>Общие фрагменты траекторий</h2><div class="cards">${fragments.slice(0,40).map((f,i)=>`<button class="card" data-fragment-card="${i}">${miniMap(f.points)}<b>${f.ids.length} эпизодов</b><small>${f.count} проходов · ${f.ids.slice(0,3).join(', ')}</small></button>`).join('')||'<div class="empty">Нет общих фрагментов для этой длины. Попробуйте уменьшить длину фрагмента.</div>'}</div>`;
 $$('[data-fragment-card]').forEach(b=>b.onclick=()=>{state.fragment=fragments[+b.dataset.fragmentCard].key;map.zoom=1;renderMap();renderAnalysis();});if($('#backFragments'))$('#backFragments').onclick=()=>{state.fragment=null;renderMap();renderAnalysis();};$$('[data-example]').forEach(b=>b.onclick=()=>{let x=chosen.examples[+b.dataset.example];state.track=x.id;$('#trackSelect').value=x.id;state.allPaths=false;state.camera=0;renderVideos();map.zoom=1;layer('where');goto(x.start);});}
function renderTime(){$('#time').value=state.time;$('#clock').textContent=clock(state.time);$('#elapsed').textContent=elapsed(state.time)+' / '+elapsed(M.end);}
function drawActivity(){
 let c=$('#activity'),ratio=devicePixelRatio||1,width=c.getBoundingClientRect().width,height=32;c.width=width*ratio;c.height=height*ratio;c.style.height=height+'px';let ctx=c.getContext('2d');ctx.scale(ratio,ratio);
 let values=smoothActivity(Array.from({length:360},(_,i)=>new Set(M.snapshot(i*M.end/359).map(o=>o[3])).size)),max=Math.max(1,...values),points=values.map((v,i)=>[i*width/(values.length-1),height-2-v/max*(height-6)]);
 ctx.beginPath();ctx.moveTo(0,height);ctx.lineTo(...points[0]);for(let i=1;i<points.length;i++){let a=points[i-1],b=points[i];ctx.bezierCurveTo((a[0]+b[0])/2,a[1],(a[0]+b[0])/2,b[1],...b);}ctx.lineTo(width,height);ctx.closePath();let gradient=ctx.createLinearGradient(0,0,0,height);gradient.addColorStop(0,'#3b82f680');gradient.addColorStop(1,'#3b82f610');ctx.fillStyle=gradient;ctx.fill();
}

function tick(now){if(state.playing&&last){state.time=Math.min(M.end,state.time+(now-last)/1000*state.speed);if(state.time>=M.end){state.playing=false;$('#play').textContent='▶';syncVideos();}}last=now;if(now-lastRender>Math.min(150,500/M.sampleHz)){if(state.playing){syncVideos();renderTime();renderMap();renderTracks();if(!$('#presence').hidden)renderGantt();}renderBoxes();lastRender=now;}requestAnimationFrame(tick);}
async function init(){try{const responses=await Promise.all(['dataset.json','floor/tracks-floor.json','floor/plan.svg','floor/calibration.json'].map(p=>fetch(p)));if(responses.some(r=>!r.ok))throw Error('Не найдены файлы данных');D=await responses[0].json();raw=(await responses[1].json()).tracks;if(D.manifest.spatial_available===false)raw={};let plan=await responses[2].text(),cfg=await responses[3].json();M=makeModel(D);state.time=Math.min(state.time,M.end);const date = D.manifest.common_t0?.slice(0, 10).split('-').reverse().join('.');
 const source = D.manifest.profile ? 'Модель · ' + D.manifest.profile : 'Эталон';
 $('#datasetDate').textContent = [date, source, M.sampleHz + ' Гц'].filter(Boolean).join(' · ');displayIntervals=D.manifest.source_format==='predictions'?D.intervals:visibilityIntervals(D.observations);status=Object.fromEntries(D.tracks.map(t=>[t.track_id,t.link_status]));map=new OfficeMap($('#map'),plan,cfg);rebuildSpatial();$('#total').textContent=D.tracks.length;$('#trackSelect').innerHTML='<option value="all">Все ID</option>'+D.tracks.map(t=>`<option>${esc(t.track_id)}</option>`).join('');$('#trackSelect').value='all';$('#trackSelect').onchange=e=>{if(e.target.value==='all'){state.track=null;renderMap();renderTracks();renderGantt();renderControls();renderBoxes();}else choose(e.target.value,true);};$('#time').max=M.end;$('#time').value=state.time;$('#ticks').innerHTML=Array.from({length:7},(_,i)=>`<span>${elapsed(M.end*i/6)}</span>`).join('');$('#time').oninput=e=>goto(+e.target.value);$('#search').oninput=renderTracks;$('#visibleOnly').onchange=renderTracks;$('#allCameras').onclick=()=>chooseCamera(0);$$('[data-layer]').forEach(b=>b.onclick=()=>layer(b.dataset.layer));$('#play').onclick=()=>{if(state.time>=M.end)goto(0);state.playing=!state.playing;$('#play').textContent=state.playing?'Ⅱ':'▶';last=performance.now();syncVideos();};$('#speed').onchange=e=>{state.speed=+e.target.value;syncVideos();};$('#start').onclick=()=>goto(0);$('#end').onclick=()=>goto(M.end);$('#topView').onclick=()=>setView(false);$('#isoView').onclick=()=>setView(true);function setView(iso){map.iso=iso;$('#topView').classList.toggle('active',!iso);$('#isoView').classList.toggle('active',iso);renderMap();}
 $('#zoomIn').onclick=()=>{map.zoom=Math.min(2.5,map.zoom*1.2);renderMap();};$('#zoomOut').onclick=()=>{map.zoom=Math.max(.65,map.zoom/1.2);renderMap();};
 $('#map').addEventListener('pointermove',e=>{if(state.layer!=='why')return;let p=mapPoint(e),cell=cells.reduce((a,c)=>!a||Math.hypot(c.x-p[0],c.y-p[1])<Math.hypot(a.x-p[0],a.y-p[1])?c:a,null);if(!cell||Math.hypot(cell.x-p[0],cell.y-p[1])>15){hideTip();return;}let t=cell.time??state.time;tip(e,`<b>${esc(map.contextAt(cell.x,cell.y))}</b><small>${clock(Math.max(0,t-state.window))} — ${clock(t)}</small><small>Индекс концентрации ${cell.v.toFixed(2)}</small><small>Аномальность: нет исторической нормы</small>`);});$('#map').onpointerleave=hideTip;
 document.onkeydown=e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key===' '){e.preventDefault();$('#play').click();}if(e.key==='ArrowRight')goto(state.time+5);if(e.key==='ArrowLeft')goto(state.time-5);};
 renderVideos();renderTracks();renderControls();renderTime();renderGantt();renderMap();workspace.activateLayer('where');drawActivity();window.addEventListener('resize',drawActivity);requestAnimationFrame(tick);
 }catch(err){console.error(err);fail('Ошибка загрузки: '+err.message+'. Запускайте приложение через python -m src.web.server.');}}
init();
