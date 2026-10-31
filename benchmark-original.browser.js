// Requires original-cuda references generated independently on NVIDIA.
export async function benchmarkOriginal(repeats=3){
 const runs=[],warmups=[];
 await window.releaseModels();
 for(const profile of ['fast','detail']){
  for(const steps of [1,2,3]){
   warmups.push(await window.runTest(true,'sample-backyard',null,profile,'wgsl',steps,true));
   for(let repeat=0;repeat<repeats;repeat++){
    for(const photo of ['sample-backyard','brick-ranch','raised-deck']){
     runs.push({repeat,...await window.runTest(true,photo,null,profile,'wgsl',steps,true)});
    }
   }
  }
 }
 return {warmups,runs};
}
