import {VerifiedModelCache,ModelPackage,validatePackage} from './model-cache.js';
import {downloadModel} from './download.js';
import {BundledModelPackage} from './model-bundles.js';
import {sha256} from './model-cache.js';
import {roundWeight24} from './weight-precision.js';
import {ProgressiveModelPackage} from './progressive-package.js';
let screen24=false,screenPlan;
let weightMode=__MODEL_PACKAGE_FORMAT__===3?'compact':'exact';
export function setWeightMode(mode){if(!['compact','exact'].includes(mode)||(__MODEL_PACKAGE_FORMAT__!==3&&mode!=='exact'))throw Error('Unavailable weight download mode');weightMode=mode;screen24=false;}
export function setWeightScreen(enabled){if(!import.meta.env.DEV&&import.meta.env.VITE_WEIGHT_SCREEN!=='1')throw Error('Weight screening is a local experiment');screen24=Boolean(enabled);}
const base=(import.meta.env.VITE_MODEL_PACKAGE_BASE_URL||__MODEL_PACKAGE_DEFAULT_BASE__).replace(/\/$/,'');
const persistent=new VerifiedModelCache();let pending;
async function getPackage(){
 if(!__MODEL_PACKAGE_ENABLED__)return null;
 if(!pending)pending=(async()=>{
  const manifest=await persistent.read(__MODEL_PACKAGE_MANIFEST_URL__,{sha256:__MODEL_PACKAGE_SHA256__,bytes:__MODEL_PACKAGE_BYTES__},async bytes=>validatePackage(JSON.parse(new TextDecoder().decode(bytes))));
  return manifest.format===3?new ProgressiveModelPackage(manifest,base,persistent):manifest.format===2?new BundledModelPackage(manifest,base,persistent):new ModelPackage(manifest,base,persistent);
 })().catch(error=>{pending=null;throw error;});
 return pending;
}
export async function packagedModel(path,fallbackUrl,bytes,progress=()=>{}){
 const pack=await getPackage();const data=await (pack?pack.load(path,progress,weightMode):downloadModel(fallbackUrl,bytes,progress));
 if(screen24){
  if(!screenPlan)screenPlan=fetch('/models/precision/fp24-screen/plan.json').then(async r=>{if(!r.ok)throw Error('Missing FP24 screening plan');return r.json();});
  const plan=(await screenPlan).files[path];
  if(plan){if(data.length!==plan.bytes||await sha256(data)!==plan.sourceSha256)throw Error('FP24 screen source mismatch');roundWeight24(data,plan.spans);if(await sha256(data)!==plan.compactSha256)throw Error('FP24 screen result mismatch');}
 }
 return data;
}
export const modelDownloadStats=()=>({packaged:__MODEL_PACKAGE_ENABLED__,progressive:__MODEL_PACKAGE_FORMAT__===3,weightMode,screen24,packageSha256:__MODEL_PACKAGE_SHA256__,...persistent.stats});
export async function clearModelCache(){await persistent.clear();pending=null;return modelDownloadStats();}
