#!/usr/bin/env node
import {cp, mkdir, realpath} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {dirname, resolve, relative, sep} from 'node:path';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export async function copyAssets(destination='public/moge-sdk') {
  const target=resolve(destination),source=resolve(root,'dist');
  await mkdir(target,{recursive:true});
  const actual=await realpath(target),inside=relative(source,actual);
  if(!inside||(!inside.startsWith(`..${sep}`)&&inside!=='..'))throw Error('The destination must be outside the SDK runtime directory');
  if(actual===resolve('.')||actual===resolve('/'))throw Error('Choose a dedicated directory such as public/moge-sdk');
  // Preserve unrelated files. Copy only the SDK runtime and license paths.
  await cp(source,target,{recursive:true});
  await cp(resolve(root,'LICENSE'),resolve(target,'LICENSE'));
  await cp(resolve(root,'licenses'),resolve(target,'licenses'),{recursive:true});
  return target;
}
if(process.argv[1]&&await realpath(process.argv[1])===fileURLToPath(import.meta.url)) {
  const args=process.argv.slice(2);
  if(args.length>1||args.some(a=>a.startsWith('-'))) {
    console.log('Usage: moge3-copy-assets [destination]\nDefault: public/moge-sdk');
    if(!args.includes('--help'))process.exitCode=1;
  }else copyAssets(args[0]).then(path=>console.log(`MoGe runtime assets copied to ${path}`)).catch(error=>{console.error(error.message);process.exitCode=1;});
}
