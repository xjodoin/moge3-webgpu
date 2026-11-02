import {sparseConfig,selectedSparseConfig} from './sparse-kernels.js';
export const AUTOTUNE_VERSION='sparse-fp32-v1';
export function rowBucket(rows){if(!Number.isSafeInteger(rows)||rows<1)throw Error('Invalid voxel count');return 2**Math.ceil(Math.log2(rows));}
export function candidatesFor(channels,subgroups,baseline){
 const values=[{id:'baseline',config:sparseConfig(baseline)}];
 const add=(id,value)=>{const config=sparseConfig(value);if(channels%config.cols||(!config.baseline&&channels%config.k))return;if(values.some(v=>JSON.stringify(v.config)===JSON.stringify(config)))return;values.push({id,config});};
 if(subgroups){for(const rows of [4,8,16])add(`broadcast-${rows}x${Math.min(channels,128)}`,{kind:'subgroup-broadcast',rows,cols:Math.min(channels,128),k:32});add('shared',selectedSparseConfig(channels,'optimized'));}
 else{add('shared-32',32);add('shared-64',64);}
 return values.slice(0,4);
}
export function median(values){const sorted=[...values].sort((a,b)=>a-b);const n=sorted.length;return n%2?sorted[n>>1]:(sorted[n/2-1]+sorted[n/2])/2;}
export function selectWinner(rows){
 const valid=rows.filter(r=>r.exact&&r.samples.length>=5&&r.samples.every(x=>Number.isFinite(x)&&x>0));
 const baseline=valid.find(r=>r.id==='baseline');if(!baseline)return 'baseline';
 const center=median(baseline.samples),noise=median(baseline.samples.map(x=>Math.abs(x-center)));
 let winner=baseline,best=center;
 for(const row of valid){const value=median(row.samples),spread=median(row.samples.map(x=>Math.abs(x-value)));if(value<best&&center-value>=Math.max(center*.08,.025,2*(noise+spread))){winner=row;best=value;}}
 return winner.id;
}
export class DecisionCache{
 constructor(storage=null,{limit=128,now=()=>Date.now()}={}){this.storage=storage;this.limit=limit;this.now=now;this.entries=new Map();this.unavailable=false;try{const raw=storage?.getItem(AUTOTUNE_VERSION);if(raw&&raw.length<131072){const rows=JSON.parse(raw);if(Array.isArray(rows))for(const row of rows.slice(-limit))if(this.valid(row))this.entries.set(row.key,row);}}catch{this.unavailable=true;}}
 valid(row){return row&&typeof row.key==='string'&&/^[a-f0-9]{64}$/.test(row.key)&&typeof row.id==='string'&&row.id.length<80&&Number.isFinite(row.time)&&row.time<=this.now()&&this.now()-row.time<30*86400000;}
 get(key,allowed){const row=this.entries.get(key);if(!this.valid(row)||!allowed.includes(row.id)){this.entries.delete(key);return null;}return row.id;}
 set(key,id){const row={key,id,time:this.now()};if(!this.valid(row))throw Error('Invalid tuning cache entry');this.entries.delete(key);this.entries.set(key,row);while(this.entries.size>this.limit)this.entries.delete(this.entries.keys().next().value);this.save();}
 save(){try{this.storage?.setItem(AUTOTUNE_VERSION,JSON.stringify([...this.entries.values()]));}catch{this.unavailable=true;}}
 delete(key){this.entries.delete(key);this.save();}
 clear(){this.entries.clear();try{this.storage?.removeItem(AUTOTUNE_VERSION);}catch{this.unavailable=true;}}
}
