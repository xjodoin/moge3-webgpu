import {defineConfig} from 'vite';
import {cpSync,existsSync,mkdirSync,writeFileSync,constants} from 'node:fs';
const headers={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};
export default defineConfig(({command})=>({
 publicDir:command==='build'?false:'public',
 server:{headers},preview:{headers},
 plugins:[{name:'release-models',closeBundle(){
  if(command!=='build')return;
  const modelBaseUrl=process.env.VITE_MODEL_BASE_URL||'/models';
  mkdirSync('dist',{recursive:true});writeFileSync('dist/model-source.json',JSON.stringify({modelBaseUrl}));
  for(const part of ['profiles/fast','profiles/detail','refiner']){
   const source=`public/models/${part}`,target=`dist/models/${part}`;
   if(existsSync(source)){mkdirSync(target,{recursive:true});cpSync(source,target,{recursive:true,mode:constants.COPYFILE_FICLONE,filter:path=>modelBaseUrl==='/models'||!path.endsWith('.onnx')});}
  }
 }}],
}));
