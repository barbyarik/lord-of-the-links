// Build visibility from actual samples, independently of local-ID segmentation.
export function visibilityIntervals(observations){
 const groups=new Map();
 for(let o of observations){if(!Number.isFinite(o[1]))continue;let key=o[0]+':'+o[3];if(!groups.has(key))groups.set(key,new Map());groups.get(key).set(o[9],o);}
 let out=[];
 for(let entries of groups.values()){
  let rows=[...entries.values()].sort((a,b)=>a[9]-b[9]),run=null,prev=null;
  for(let o of rows){if(!prev||o[9]-prev[9]>500||o[2]-prev[2]>.8||o[1]<prev[1]){if(run)out.push(run);run={track_id:o[3],camera_id:o[0],first_seen_s:o[1],last_seen_s:o[1],observation_count:1};}else{run.last_seen_s=o[1];run.observation_count++;}prev=o;}
  if(run)out.push(run);
 }
 return out;
}
export function rowLayout(lanes,bar=21,gap=6,padding=10){let count=Math.max(1,lanes);return{height:2*padding+count*bar+(count-1)*gap,offset:padding,stride:bar+gap};}
export function smoothActivity(values){return values.map((_,i)=>{let sum=0,w=0;for(let d=-3;d<=3;d++){let j=Math.max(0,Math.min(values.length-1,i+d)),weight=Math.exp(-d*d/3);sum+=values[j]*weight;w+=weight;}return sum/w;});}
