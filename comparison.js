// Fixed release gates: <0.1% NRMSE; mask disagreements may only be confidence
// values within 1e-5 of 0.5 and at most one scalar in 100,000. NaNs always fail.
export function compare(actual,expected,{boundaryMask}={}){
 if(actual.length!==expected.length)throw Error('Output shape mismatch');
 let diff=0,scale=0,maxAbs=0,finite=true,maskMismatches=0,unexplainedMaskMismatches=0;
 const errors=[];
 for(let i=0;i<actual.length;i++){
  if(!Number.isFinite(actual[i])||!Number.isFinite(expected[i])){
   if(actual[i]===Infinity&&expected[i]===Infinity)continue;
   if(Number.isNaN(actual[i])||Number.isNaN(expected[i])||actual[i]===-Infinity||expected[i]===-Infinity){finite=false;continue;}
   maskMismatches++;
   const confidence=boundaryMask?.[Math.floor(i/3)];
   if(confidence===undefined||Math.abs(confidence-.5)>1e-5)unexplainedMaskMismatches++;
   continue;
  }
  const d=actual[i]-expected[i],error=Math.abs(d);diff+=d*d;scale+=expected[i]*expected[i];maxAbs=Math.max(maxAbs,error);errors.push(error);
 }
 errors.sort((a,b)=>a-b);
 return {finite,maxAbs,nrmse:Math.sqrt(diff/Math.max(scale,1e-20)),p99Abs:errors[Math.floor(errors.length*.99)]??0,
  maskMismatches,maskMismatchFraction:maskMismatches/actual.length,unexplainedMaskMismatches};
}
export const passes=r=>r.finite&&r.nrmse<.001&&r.maskMismatchFraction<=.00001&&r.unexplainedMaskMismatches===0;
