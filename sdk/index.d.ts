export interface Reconstruction {
 width:number; height:number; points:Float32Array; depth:Float32Array; mask:Uint8Array; colors:Uint8Array;
 /** Row-major 3x3 camera matrix, normalized to the cropped input. OpenCV axes. */
 intrinsics:Float32Array;
 imageTransform:{originalWidth:number;originalHeight:number;crop:[number,number,number,number]};
 profile:'fast'|'detail';passes:1|2|3;weightMode:'exact'|'compact';measurementStatus:'estimated';
}
export interface ReconstructionOptions {profile?:'fast'|'detail';passes?:1|2|3;weightMode?:'exact'|'compact';signal?:AbortSignal;onProgress?:(message:string)=>void;}
export class MoGeClient {
 constructor(options?:{workerUrl?:string|URL});
 static isSupported():Promise<boolean>;
 reconstruct(photo:Blob,options?:ReconstructionOptions):Promise<Reconstruction>;
 dispose():void;
}
