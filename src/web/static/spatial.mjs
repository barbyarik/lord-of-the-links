import {nearest} from './core.mjs';
// Shared spatial observations drive all three layers. Distances are plan units.
export function cleanTracks(raw,exclude=false){return Object.fromEntries(Object.entries(raw).map(([id,rows])=>[id,rows.filter(p=>!exclude||p.mode!=='largest_box_disagreement').sort((a,b)=>a.t_s-b.t_s)]));}
export function currentPoints(tracks, time, camera = 0, sampleHz = 2) {
  const points = [];
  for (const [id, rows] of Object.entries(tracks)) {
    const point = nearest(rows, time, 't_s');
    if (point && Math.abs(point.t_s - time) <= .502 / sampleHz &&
        (!camera || point.sources.some(source => source.camera_id === camera))) {
      points.push({...point, id});
    }
  }
  return points;
}
export function splitRuns(rows,maxGap=1.01,maxStep=65){let runs=[],run=[];for(let p of rows){let prev=run.at(-1);if(prev&&(p.segment!==prev.segment||p.t_s-prev.t_s>maxGap||Math.hypot(p.x-prev.x,p.y-prev.y)>maxStep)){if(run.length)runs.push(run);run=[];}run.push(p);}if(run.length)runs.push(run);return runs;}
export function frequentFragments(tracks,status,length=3,cell=18){let groups=new Map();for(let[id,rows]of Object.entries(tracks)){if(!['linked_visual','prediction'].includes(status[id]))continue;for(let run of splitRuns(rows)){let visits=[];for(let p of run){let key=Math.floor(p.x/cell)+','+Math.floor(p.y/cell),prev=visits.at(-1);if(prev?.key===key){prev.end=p;for(let source of p.sources)prev.cameras.add(source.camera_id);continue;}visits.push({key,start:p,end:p,cameras:new Set(p.sources.map(s=>s.camera_id))});}for(let i=0;i+length<=visits.length;i++){let part=visits.slice(i,i+length),key=part.map(v=>v.key).join('|');if(!groups.has(key))groups.set(key,{key,points:part.map(v=>[v.start.x,v.start.y]),examples:[]});let cameras=[...new Set(part.flatMap(v=>[...v.cameras]))].sort();groups.get(key).examples.push({id,start:part[0].start.t_s,end:part.at(-1).end.t_s,cameras,points:part.flatMap(v=>[[v.start.x,v.start.y],[v.end.x,v.end.y]])});}}}
 return [...groups.values()].map(g=>({...g,count:g.examples.length,ids:[...new Set(g.examples.map(e=>e.id))]})).filter(g=>g.ids.length>=2).sort((a,b)=>b.ids.length-a.ids.length||b.count-a.count);
}
export function densityGrid(tracks,t,window=5,cell=8,radius=22,sampleHz=2){let ticks=new Map();for(let[id,rows]of Object.entries(tracks))for(let p of rows){if(p.t_s>t||p.t_s<=t-window)continue;let key=Math.round(p.t_s*sampleHz);if(!ticks.has(key))ticks.set(key,[]);ticks.get(key).push({...p,id});}let sums=new Map();for(let points of ticks.values()){let frame=new Map();for(let p of points){for(let x=Math.floor((p.x-radius)/cell);x<=Math.ceil((p.x+radius)/cell);x++)for(let y=Math.floor((p.y-radius)/cell);y<=Math.ceil((p.y+radius)/cell);y++){let cx=(x+.5)*cell,cy=(y+.5)*cell,d=Math.hypot(p.x-cx,p.y-cy);if(d>radius)continue;let key=x+','+y;frame.set(key,(frame.get(key)||0)+Math.exp(-d*d/(2*(radius/2)**2)));}}for(let[key,v]of frame)sums.set(key,Math.max(sums.get(key)||0,v));}return [...sums].map(([key,v])=>{let[x,y]=key.split(',').map(Number);return{key,x:(x+.5)*cell,y:(y+.5)*cell,v,size:cell};});}
export function densitySummary(tracks,end,window=5,radius=22,sampleHz=2){
 let samples=[],all=new Map();for(let t=0;t<=end;t+=.5){let cells=densityGrid(tracks,t,window,8,radius,sampleHz);if(cells.length){let p=cells.reduce((a,b)=>a.v>b.v?a:b);samples.push({...p,time:t});for(let c of cells)if(!all.has(c.key)||all.get(c.key).v<c.v)all.set(c.key,{...c,time:t});}}
 samples.sort((a,b)=>b.v-a.v);let peaks=[];for(let p of samples){if(!peaks.some(q=>Math.abs(q.time-p.time)<20&&Math.hypot(q.x-p.x,q.y-p.y)<35))peaks.push(p);if(peaks.length===12)break;}return{peaks,cells:[...all.values()]};
}
