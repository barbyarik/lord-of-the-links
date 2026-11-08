// Door and furnishing positions traced against office.jpg in the existing plan frame.
// Keep this geometry shared by the top and isometric views; floor tracks are unchanged.
export const doors=[
 {h:[153,129],c:[153,144],o:[138,129]},
 {h:[195,129],c:[207,129],o:[195,117]},
 {h:[223,129],c:[209,129],o:[223,115]},
 {h:[153,184],c:[153,171],o:[140,184]},
 {h:[285,217],c:[285,234],o:[268,217]},
 {h:[156,360],c:[143,360],o:[156,347]},
 {h:[191,345],c:[205,345],o:[191,331]},
 {h:[241,345],c:[255,345],o:[241,331]},
 {h:[188,384],c:[188,373],o:[177,384]},
 {h:[188,418],c:[188,431],o:[201,418]},
 {h:[284,516],c:[284,532],o:[268,516]},
 {h:[258,549],c:[258,562],o:[245,549]},
 {h:[385,546],c:[385,531],o:[370,546]},
 {h:[449,489],c:[464,489],o:[449,504]},
 {h:[470,489],c:[456,489],o:[470,475]},
 {h:[533,489],c:[519,489],o:[533,475]},
];
export function axisSegments(d){let tokens=d.match(/[MLHVZmlhvz]|-?\d*\.?\d+/g)||[],i=0,cmd='',p=[0,0],first=null,out=[];while(i<tokens.length){if(/[a-z]/i.test(tokens[i]))cmd=tokens[i++];let rel=cmd===cmd.toLowerCase(),C=cmd.toUpperCase(),q;if(C==='Z'){q=first;cmd='';}else if(C==='M'||C==='L'){let a=+tokens[i++],b=+tokens[i++];q=rel?[p[0]+a,p[1]+b]:[a,b];if(C==='M'){p=q;first=q;cmd=rel?'l':'L';continue;}}else if(C==='H'){let v=+tokens[i++];q=[rel?p[0]+v:v,p[1]];}else if(C==='V'){let v=+tokens[i++];q=[p[0],rel?p[1]+v:v];}else throw Error('Unsupported wall command '+cmd);if(q){out.push([p,q]);p=q;}}return out;}
export function cutDoors(segments,doorMarkers=doors){let out=[];for(let[a,b]of segments){let horizontal=Math.abs(a[1]-b[1])<.1,vertical=Math.abs(a[0]-b[0])<.1;if(!horizontal&&!vertical){out.push([a,b]);continue;}let axis=horizontal?0:1,fixed=1-axis,start=Math.min(a[axis],b[axis]),end=Math.max(a[axis],b[axis]),ranges=[[start,end]];for(let d of doorMarkers){if(Math.abs(d.h[fixed]-a[fixed])>2||Math.abs(d.c[fixed]-a[fixed])>2)continue;let lo=Math.min(d.h[axis],d.c[axis])-.5,hi=Math.max(d.h[axis],d.c[axis])+.5;if(hi-lo<5)continue;let next=[];for(let[l,r]of ranges){if(hi<=l||lo>=r)next.push([l,r]);else{if(lo>l)next.push([l,lo]);if(hi<r)next.push([hi,r]);}}ranges.splice(0,ranges.length,...next);}for(let[l,r]of ranges){let p=[...a],q=[...a];p[axis]=l;q[axis]=r;out.push([p,q]);}}return out;}
export function preparePlan(root,doorMarkers=doors){let walls=[];for(let p of [...root.querySelectorAll('path')]){let stroke=p.getAttribute('stroke');if(['#334155','#64748b'].includes(stroke)){walls.push(...axisSegments(p.getAttribute('d')));p.remove();}else if(stroke==='#94a3b8'&&p.getAttribute('d').includes('a15'))p.remove();}
 let segments=cutDoors(walls,doorMarkers),path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d',segments.map(([a,b])=>`M${a}L${b}`).join(' '));path.setAttribute('fill','none');path.setAttribute('stroke','#475569');path.setAttribute('stroke-width','3.2');path.setAttribute('stroke-linejoin','round');root.append(path);
 let arc=document.createElementNS(path.namespaceURI,'path');arc.setAttribute('d',doorMarkers.filter(d=>Math.hypot(d.c[0]-d.h[0],d.c[1]-d.h[1])>=7).map(d=>{let r=Math.hypot(d.o[0]-d.h[0],d.o[1]-d.h[1]),cross=(d.o[0]-d.h[0])*(d.c[1]-d.h[1])-(d.o[1]-d.h[1])*(d.c[0]-d.h[0]);return `M${d.h}L${d.o} M${d.o}A${r},${r} 0 0 ${cross>0?1:0} ${d.c}`;}).join(' '));arc.setAttribute('fill','none');arc.setAttribute('stroke','#9aaabd');arc.setAttribute('stroke-width','1.1');root.append(arc);return segments;}
export const observedAreas=[[[153,129],[319,129],[319,205],[153,205]],[[285,129],[323,129],[323,565],[285,565]],[[157,344],[287,344],[287,373],[157,373]]];
export function observedAt(x, y, margin = 0, areas = observedAreas) {
  return areas.some(area => {
    const xs = area.map(point => point[0]);
    const ys = area.map(point => point[1]);
    return x >= Math.min(...xs) - margin && x <= Math.max(...xs) + margin &&
      y >= Math.min(...ys) - margin && y <= Math.max(...ys) + margin;
  });
}
export const furniture=[
 {x:198,y:157,w:46,h:28,type:'pingpong'},
 {x:245,y:194,w:36,h:10,type:'tv'},
 {x:247,y:143,w:59,h:11,type:'counter'},
 {x:162,y:185,w:25,h:9,type:'bench'},
 {x:301,y:135,w:7,h:7,type:'plant'},
 {x:274,y:190,w:7,h:7,type:'plant'}
];

export const cameras=[{id:1,name:'Игровая зона',p:[231,129],fov:[[231,129],[153,205],[319,180]]},{id:2,name:'Обратный вид',p:[153,205],fov:[[153,205],[181,129],[319,129],[319,180]]},{id:3,name:'Длинный коридор',p:[309,129],fov:[[309,129],[285,205],[285,565],[323,565]]},{id:4,name:'Поперечный проход',p:[323,359],fov:[[323,359],[157,344],[157,373]]},{id:5,name:'Коридор у 342',p:[301,550],fov:[[301,550],[287,129],[323,129]]}];

// Optional display-only decoration for the original office.
export const officeLayout = {
  cameras, doors, furniture, observedAreas,
  center: [339, 345],
  topSize: [572, 620],
  isoSize: [910, 600],
  outline: [[68,95],[195,95],[195,129],[223,129],[223,47],[323,47],
            [323,456],[610,456],[610,630],[68,630]],
  areas: [
    {bounds: [153, null, 285, 205], label: 'Игровая зона · теннисный стол'},
    {bounds: [null, 344, 285, 374], label: 'Поперечный проход · двери комнат'},
    {bounds: [null, 510, 330, null], label: 'Коридор у комнаты 342'},
    {bounds: [323, 450, null, null], label: 'Поворот коридора · лестница'},
  ],
  defaultArea: 'Длинный коридор · двери кабинетов',
};
