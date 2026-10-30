import {defineConfig} from 'vite';
import {cpSync,existsSync,mkdirSync,constants} from 'node:fs';
const headers={'Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'};
export default defineConfig(({command})=>({
 publicDir:command==='build'?false:'public',
 server:{headers},preview:{headers},
 plugins:[{name:'release-models',closeBundle(){
  if(command!=='build')return;
  for(const part of ['profiles/fast','profiles/detail','refiner']){
   const source=`public/models/${part}`,target=`dist/models/${part}`;
   if(existsSync(source)){mkdirSync(target,{recursive:true});cpSync(source,target,{recursive:true,mode:constants.COPYFILE_FICLONE});}
  }
 }}],
}));
