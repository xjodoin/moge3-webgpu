// Requires Playwright; set PLAYWRIGHT_MODULE when it is installed outside this checkout.
const {chromium,firefox}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const fs=require('node:fs/promises');
(async()=>{
 const name=process.env.BROWSER||'chrome';
 const context=await (name==='firefox'?firefox:chromium).launchPersistentContext(`results/${name}-benchmark-profile`,name==='firefox'?{headless:false,firefoxUserPrefs:{'dom.webgpu.enabled':true}}:{channel:'chrome',headless:false,args:['--disable-background-timer-throttling','--disable-renderer-backgrounding']});
 const page=context.pages()[0];await page.route('**/favicon.ico',route=>route.fulfill({status:204}));const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 try{
 await page.goto(process.env.TEST_URL||'http://127.0.0.1:5181');await page.waitForFunction(()=>!!window.runTest);
 const photos=process.env.REAL==='1'?JSON.parse(await fs.readFile('fixtures/real/manifest.json','utf8')).map(r=>r.id):undefined;
 const options={...(photos?{photos,referenceScope:'original-only'}:{}),repeats:Number(process.env.REPEATS||3)};
 await page.evaluate(({timing,options})=>{window.benchmarkJob=window.runAutotuneBenchmark(timing,options).catch(e=>window.benchmarkError=String(e));},{timing:process.env.TIMING==='1',options});
 let report;for(;;){await page.waitForTimeout(10000);const status=await page.evaluate(()=>({progress:window.autotuneProgress,complete:window.autotuneReport?.complete,error:window.benchmarkError,runs:window.autotuneReport?.runs.length}));console.log(JSON.stringify(status));if(status.complete||status.error){report=await page.evaluate(()=>window.autotuneReport);report.error=status.error;break;}}
 report.browserErrors=errors;
 if(!report.timing)report.runs=report.runs.map(r=>Object.fromEntries(['profile','caseName','refineSteps','repeat','enabled','passed','validated','validationReference','referenceScope','original','outputs','computeMs','inputShape'].map(k=>[k,r[k]])));
 await fs.writeFile(process.env.REPORT_FILE||'benchmarks/browser-report.json',JSON.stringify(report,null,2)+'\n');
 if(report.error||errors.length||!report.complete||report.passed===false)throw Error(report.error||errors.join('\n')||(report.passed===false?'CUDA accuracy failures; see saved report':'Incomplete benchmark'));
 }finally{await context.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
