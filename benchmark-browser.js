// Import in the browser console after loading the app. Run MLX separately.
export async function benchmark(repeats=3,sparseBackend='wgsl') {
  const warmups=[],runs=[];
  await window.releaseModels();
  for(const profile of ['fast','detail']) {
    warmups.push(await window.runTest(true,'sample-backyard',null,profile,sparseBackend));
    for(let repeat=0;repeat<repeats;repeat++) {
      for(const photo of ['sample-backyard','brick-ranch','raised-deck']) {
        runs.push({repeat,...await window.runTest(true,photo,null,profile,sparseBackend)});
      }
    }
  }
  return {warmups,runs};
}
