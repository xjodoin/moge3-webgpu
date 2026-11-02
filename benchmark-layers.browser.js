// Local experiment. Reports sensitivity, never promotes a package automatically.
export async function benchmarkLayers({names=null,bits=[20,24],profiles=['fast','detail'],photos=['sample-backyard','brick-ranch','raised-deck'],steps=[1,2,3]}={}){
 const response=await fetch('/models/precision/fp24-screen/plan.json');if(!response.ok)throw Error('Missing verified screening plan');
 const plan=await response.json();
 const available=plan.files['profiles/fast/moge3-dense.onnx'].spans;
 const selected=names??available.map(s=>s.name);
 if(selected.some(name=>!available.some(s=>s.name===name)))throw Error('Unknown initializer');
 const report={started:Date.now(),userAgent:navigator.userAgent,policy:'isolated-layer-storage-only',runs:[],candidates:[],complete:false};
 window.layerReport=report;
 const previous=window.modelDownloadStats().weightMode;
 try{
  await window.setWeightMode('exact');await window.setWeightScreen(false);
  for(const name of selected)for(const width of bits){
   await window.setLayerScreen([{name,bits:width}]);
   const candidate={name,bits:width,passed:true,runs:[]};report.candidates.push(candidate);
   for(const profile of profiles)for(const pass of steps)for(const photo of photos){
    window.layerProgress={name,bits:width,profile,pass,photo};
    const row=await window.runTest(true,photo,null,profile,'wgsl',pass,true);
    candidate.runs.push(report.runs.length);report.runs.push(row);
    candidate.passed&&=row.passed===true;
   }
  }
  report.complete=true;return report;
 }finally{await window.setLayerScreen(null);await window.setWeightMode(previous);report.finished=Date.now();}
}
