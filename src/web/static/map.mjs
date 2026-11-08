import {preparePlan,observedAt,officeLayout} from './environment.mjs';
import {splitRuns} from './spatial.mjs';
const NS='http://www.w3.org/2000/svg';
const colors=['#2563eb','#0ea5e9','#6366f1','#1d4ed8','#64748b'];
export const color=id=>`hsl(${Math.abs([...id].reduce((s,c)=>(s*31+c.charCodeAt(0))|0,0))%75+195} 65% 43%)`;
export const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const el=(tag,a={},text='')=>`<${tag} ${Object.entries(a).map(([k,v])=>`${k}="${v}"`).join(' ')}>${text}</${tag}>`;
export class OfficeMap{
 constructor(svg, source, config) {
  this.svg = svg;
  this.iso = false;
  this.zoom = 1;
  this.walkable = config.geometry?.walkable ?? config.anchors?.walkable ?? [];
  this.layout = config.display ?? (config.anchors?.plan === 'office_coverage.svg' ? officeLayout : {});
  this.cameras = this.layout.cameras ?? [];
  const dom = new DOMParser().parseFromString(source, 'image/svg+xml');
  this.source = dom.documentElement;
  const [x, y, width, height] = config.plan_view_box ??
    this.source.getAttribute('viewBox')?.trim().split(/[ ,]+/).map(Number) ?? [0, 0, 572, 605];
  this.center = this.layout.center ?? [x + width / 2, y + height / 2];
  this.topSize = this.layout.topSize ?? [width, height];
  this.isoSize = this.layout.isoSize ?? [width + .62 * height, .35 * width + .52 * height + 35];
  this.outline = this.layout.outline ?? [[x,y],[x+width,y],[x+width,y+height],[x,y+height]];
  this.walls = this.layout.doors ? preparePlan(this.source, this.layout.doors) : [];
  const paths = [...this.source.children].filter(node => node.tagName === 'path');
  if (this.layout === officeLayout) {
   if (paths[0]) paths[0].setAttribute('fill', '#3c4655');
   if (paths[1]) paths[1].setAttribute('fill', '#485568');
  }
  for (const [index, points] of (this.layout.observedAreas ?? []).entries()) {
   const area = document.createElementNS(NS, 'polygon');
   area.setAttribute('points', points.map(point => point.join(',')).join(' '));
   area.setAttribute('fill', index === 0 ? '#c6d4e3' : '#b8c8da');
   this.source.insertBefore(area, this.source.querySelector('text'));
  }
  this.source.querySelectorAll('text').forEach(text => text.setAttribute('fill', '#98a8bd'));
  const floorNodes = [...this.source.children];
  this.floor = floorNodes.filter(node => this.layout !== officeLayout || node.tagName !== 'rect')
    .map(node => node.outerHTML).join('');
 }

 p([x,y],z=0) {
  const [cx, cy] = this.center;
  return this.iso ? [(x-cx)*.92+(y-cy)*.58,(y-cy)*.48-(x-cx)*.32-z] : [x,y];
 }

 planPoint(x, y) {
  if (!this.iso) return [x, y];
  const determinant = .92 * .48 + .58 * .32;
  return [(.48*x-.58*y)/determinant+this.center[0],(.32*x+.92*y)/determinant+this.center[1]];
 }

 contextAt(x, y) {
  const area = (this.layout.areas ?? []).find(({bounds: [left, top, right, bottom]}) =>
    (left === null || x >= left) && (top === null || y >= top) &&
    (right === null || x < right) && (bottom === null || y < bottom));
  return area?.label ?? this.layout.defaultArea ?? 'План офиса';
 }

 points(a){return a.map(p=>this.p(p).join(',')).join(' ');}
 line(points,attrs={}){return el('polyline',{points:this.points(points),fill:'none','stroke-linecap':'round','stroke-linejoin':'round',...attrs});}
 draw({tracks,current,time,track,camera,showCameras=true,layer,allPaths,trail,cells,fragments,fragment,fieldMax}){
  const [width, height] = (this.iso ? this.isoSize : this.topSize).map(value => value / this.zoom);
  const [cx, cy] = this.iso ? [0, 0] : this.center;
  this.svg.setAttribute('viewBox', [cx-width/2, cy-height/2, width, height].join(' '));
  const [originX, originY] = this.p([0, 0]);
  const isoMatrix = `matrix(.92,-.32,.58,.48,${originX},${originY})`;
  let clip=this.walkable.map(p=>this.p(p).join(',')).join(' '),s=`<defs><clipPath id="walkable"><polygon points="${clip}"/></clipPath><filter id="heatBlur"><feGaussianBlur stdDeviation="${this.iso?3:5}"/></filter><filter id="shadow"><feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#3b5147" flood-opacity=".22"/></filter></defs>`;
  if(this.iso){let outline=this.outline;s+=el('polygon',{points:outline.map(p=>this.p(p,-10).join(',')).join(' '),fill:'#24364f',filter:'url(#shadow)'});s+=el('g',{transform:isoMatrix},this.floor);let walls=this.walls.map(([a,b])=>({a,b,y:(this.p(a)[1]+this.p(b)[1])/2})).sort((a,b)=>a.y-b.y);for(let{a,b}of walls){let seen=observedAt((a[0]+b[0])/2,(a[1]+b[1])/2,5,this.layout.observedAreas ?? []),height=seen?23:13;s+=el('polygon',{points:[this.p(a),this.p(b),this.p(b,height),this.p(a,height)].map(p=>p.join(',')).join(' '),fill:seen?(b[0]>a[0]?'#e3edfa':'#bdcde0'):(b[0]>a[0]?'#67778d':'#505f75'),'fill-opacity':.87,stroke:'none'});s+=this.line([a,b],{stroke:seen?'#eff6ff':'#7f90a6','stroke-width':2,transform:`translate(0,-${height})`});}
  }else s+=this.floor;
  // The same furniture footprint is used in both views.
  for(let item of this.layout.furniture ?? []){let{x,y,w,h,type}=item,z=type==='plant'?10:type==='counter'?8:6;
   if(type==='plant'){let p=this.p([x+w/2,y+h/2]);s+=el('ellipse',{cx:p[0],cy:p[1],rx:3.5,ry:2.5,fill:'#34465e'});let q=this.p([x+w/2,y+h/2],this.iso?z:0);s+=el('circle',{cx:q[0],cy:q[1],r:4,fill:'#8aabac'});continue;}
   let corners=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],top=corners.map(p=>this.p(p,this.iso?z:0)),fill=type==='pingpong'?'#2367aa':type==='counter'?'#899fbf':type==='tv'?'#324967':'#6783a5';
   if(this.iso)for(let i of [1,2])s+=el('polygon',{points:[this.p(corners[i]),this.p(corners[(i+1)%4]),top[(i+1)%4],top[i]].map(p=>p.join(',')).join(' '),fill:'#becbdb',stroke:'#a2b3c8','stroke-width':.4});
   s+=el('polygon',{points:top.map(p=>p.join(',')).join(' '),fill,stroke:'#90a4be','stroke-width':.8});
   if(type==='pingpong'){
    let middle=[[x+w/2,y],[x+w/2,y+h]],endpoints=middle.map(p=>this.p(p,this.iso?z:0));s+=el('line',{x1:endpoints[0][0],y1:endpoints[0][1],x2:endpoints[1][0],y2:endpoints[1][1],stroke:'#dbeeff','stroke-width':1});
    if(this.iso){let net=[this.p(middle[0],z),this.p(middle[1],z),this.p(middle[1],z+4),this.p(middle[0],z+4)];s+=el('polygon',{points:net.map(p=>p.join(',')).join(' '),fill:'#ffffff36',stroke:'#dbeafe','stroke-width':.6});for(let q of corners){let a=this.p(q),b=this.p(q,z);s+=el('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#101b2e','stroke-width':1.4});}}
   }
   if(type==='tv'){
    if(this.iso){let screen=[[x+3,y+h,9],[x+w-3,y+h,9],[x+w-3,y+h,26],[x+3,y+h,26]].map(p=>this.p([p[0],p[1]],p[2]));s+=el('polygon',{points:screen.map(p=>p.join(',')).join(' '),fill:'#071426',stroke:'#75a6dd','stroke-width':1.2});let q=this.p([x+w/2,y+h],17);s+=el('text',{x:q[0],y:q[1]+1,'text-anchor':'middle','font-size':5,fill:'#7fb6ff'},'TV');}
    else{s+=el('rect',{x:x+3,y:y+1,width:w-6,height:5,rx:1,fill:'#081426',stroke:'#77a7db','stroke-width':.6});}
   }
  }
  if(showCameras&&layer==='where')this.cameras.forEach(c=>s+=el('polygon',{points:this.points(c.fov),fill:colors[(c.id-1)%colors.length],'fill-opacity':camera===c.id?.16:.045,stroke:colors[(c.id-1)%colors.length],'stroke-opacity':camera===c.id?.7:.2,'stroke-width':1}));
  if(layer==='why'){s+='<g clip-path="url(#walkable)"><g filter="url(#heatBlur)">';for(let c of cells){let p=this.p([c.x,c.y]),v=Math.min(1,c.v/Math.max(1,fieldMax));s+=el('circle',{cx:p[0],cy:p[1],r:this.iso?6:8,fill:`hsl(${55-v*50} 88% ${67-v*22}%)`,'fill-opacity':Math.max(.16,v*.9)});}s+='</g></g>';}
  if(layer==='where')for(let[id,rows]of Object.entries(tracks)){if(track&&id!==track)continue;let relevant=allPaths?rows:rows.filter(p=>p.t_s<=time&&p.t_s>=time-trail);if(camera)relevant=relevant.filter(p=>p.sources.some(s=>s.camera_id===camera));for(let run of splitRuns(relevant)){if(run.length<2)continue;s+=this.line(run.map(p=>[p.x,p.y]),{stroke:color(id),'stroke-width':track?3.4:2.1,'stroke-opacity':track?.9:.55,'data-track':escape(id),class:'interactive'});}}
  if(layer==='how')for(let f of [...fragments.slice(0,18),...fragments.filter(f=>f.key===fragment&&!fragments.slice(0,18).includes(f))]){let selected=fragment===f.key;let pts=f.points;s+=this.line(pts,{stroke:selected?'#e5f0ff':'#438cff','stroke-width':selected?8:Math.min(8,2+f.ids.length*.5),'stroke-opacity':selected?.95:.43,'data-fragment':escape(f.key),class:'interactive'});}
  if(layer==='where')for(let p of current){if(track&&p.id!==track)continue;let q=this.p([p.x,p.y]),approx=p.mode==='largest_box_disagreement';s+=`<g class="interactive" data-track="${escape(p.id)}">`+el('circle',{cx:q[0],cy:q[1],r:track?5.7:4.5,fill:color(p.id),stroke:'#fff','stroke-width':2,'stroke-dasharray':approx?'2 2':'none'})+el('text',{x:q[0]+8,y:q[1]-8,'font-size':11,fill:'#0f172a',stroke:'#fff','stroke-width':3,'paint-order':'stroke'},escape(p.id))+'</g>';}
  if(showCameras)this.cameras.forEach(c=>{let q=this.p(c.p,0);s+=`<g data-camera="${c.id}" class="interactive">`+el('circle',{cx:q[0],cy:q[1],r:camera===c.id?10:8,fill:colors[(c.id-1)%colors.length],stroke:'#fff','stroke-width':2})+el('text',{x:q[0],y:q[1]+3.5,'text-anchor':'middle','font-size':10,fill:'#fff'},c.id)+'</g>';});
  this.svg.innerHTML=s;
 }
}
