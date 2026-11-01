import {VerifiedModelCache,ModelPackage,validatePackage} from './model-cache.js';
import {downloadModel} from './download.js';
import {BundledModelPackage} from './model-bundles.js';
const base=(import.meta.env.VITE_MODEL_PACKAGE_BASE_URL||__MODEL_PACKAGE_DEFAULT_BASE__).replace(/\/$/,'');
const persistent=new VerifiedModelCache();let pending;
async function getPackage(){
 if(!__MODEL_PACKAGE_ENABLED__)return null;
 if(!pending)pending=(async()=>{
  const manifest=await persistent.read(__MODEL_PACKAGE_MANIFEST_URL__,{sha256:__MODEL_PACKAGE_SHA256__,bytes:__MODEL_PACKAGE_BYTES__},async bytes=>validatePackage(JSON.parse(new TextDecoder().decode(bytes))));
  return manifest.format===2?new BundledModelPackage(manifest,base,persistent):new ModelPackage(manifest,base,persistent);
 })().catch(error=>{pending=null;throw error;});
 return pending;
}
export async function packagedModel(path,fallbackUrl,bytes,progress=()=>{}){
 const pack=await getPackage();return pack?pack.load(path,progress):downloadModel(fallbackUrl,bytes,progress);
}
export const modelDownloadStats=()=>({packaged:__MODEL_PACKAGE_ENABLED__,packageSha256:__MODEL_PACKAGE_SHA256__,...persistent.stats});
export async function clearModelCache(){await persistent.clear();pending=null;return modelDownloadStats();}
