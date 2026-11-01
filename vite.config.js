import {defineConfig} from 'vite';
import {cpSync,existsSync,mkdirSync,writeFileSync,readFileSync,constants} from 'node:fs';
import {createHash} from 'node:crypto';
const headers={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};
export default defineConfig(({command})=>{
 const modelBaseUrl=process.env.VITE_MODEL_BASE_URL||'/models';
 const packageFolder=process.env.VITE_MODEL_PACKAGE_FORMAT!=='1'&&existsSync('public/models/packed-v2/manifest.json')?'packed-v2':'packed';
 const packageEnabled=existsSync(`public/models/${packageFolder}/manifest.json`)&&(command!=='build'||modelBaseUrl==='/models'||Boolean(process.env.VITE_MODEL_PACKAGE_BASE_URL));
 const packed=packageEnabled?readFileSync(`public/models/${packageFolder}/manifest.json`):null;
 const packageFormat=packed?JSON.parse(packed).format:null;
 return {
 define:{__MODEL_PACKAGE_ENABLED__:JSON.stringify(packageEnabled),__MODEL_PACKAGE_SHA256__:JSON.stringify(packed?createHash('sha256').update(packed).digest('hex'):''),__MODEL_PACKAGE_BYTES__:packed?.length??0,__MODEL_PACKAGE_DEFAULT_BASE__:JSON.stringify(`/models/${packageFolder}`),__MODEL_PACKAGE_MANIFEST_URL__:JSON.stringify(command==='build'?'/models/packed/manifest.json':`/models/${packageFolder}/manifest.json`)},
 publicDir:command==='build'?false:'public',
 server:{headers},preview:{headers},
 plugins:[{name:'release-models',closeBundle(){
  if(command!=='build')return;
  const modelBaseUrl=process.env.VITE_MODEL_BASE_URL||'/models';
  const sparseModelBaseUrl=process.env.VITE_SPARSE_MODEL_BASE_URL||'/models';
  const modelPackageBaseUrl=packageEnabled?(process.env.VITE_MODEL_PACKAGE_BASE_URL||`/models/${packageFolder}`):null;
  mkdirSync('dist',{recursive:true});writeFileSync('dist/model-source.json',JSON.stringify({modelBaseUrl,sparseModelBaseUrl,modelPackageBaseUrl,modelPackageFormat:packageFormat}));
  if(packageEnabled){
   mkdirSync('dist/models/packed',{recursive:true});writeFileSync('dist/models/packed/manifest.json',packed);
   if(modelPackageBaseUrl===`/models/${packageFolder}`){const part=packageFormat===2?'bundles':'chunks';mkdirSync(`dist/models/${packageFolder}`,{recursive:true});cpSync(`public/models/${packageFolder}/${part}`,`dist/models/${packageFolder}/${part}`,{recursive:true,mode:constants.COPYFILE_FICLONE});}
  }
  // Independent original-CUDA goldens used by sample validation. Trace tensors
  // stay local; they are not needed to run or validate the hosted demo.
  if(existsSync('public/original-cuda'))cpSync('public/original-cuda','dist/original-cuda',{recursive:true,mode:constants.COPYFILE_FICLONE,filter:path=>!path.split('/').includes('diagnostics')});
  for(const part of ['profiles/fast','profiles/detail','refiner','refiner-wgsl']){
   const source=`public/models/${part}`,target=`dist/models/${part}`;
   if(existsSync(source)){mkdirSync(target,{recursive:true});cpSync(source,target,{recursive:true,mode:constants.COPYFILE_FICLONE,filter:path=>!path.split('/').includes('diagnostics')&&(part==='refiner-wgsl'?((!packageEnabled&&sparseModelBaseUrl==='/models')||!path.endsWith('.bin')):((!packageEnabled&&modelBaseUrl==='/models')||!path.endsWith('.onnx')))});}
  }
 }}],
};});
