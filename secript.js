/*
  Gesture Translator Prototype
  - MediaPipe Hands for landmark detection
  - k-NN classification on normalized landmark vectors
  - Train by capturing examples per label
  - Optional translation via user-provided endpoint (JSON POST)
  - Speech synthesis for output
*/

// Utilities
const el = id => document.getElementById(id);
const video = el('video');
const canvas = el('overlay');
const ctx = canvas.getContext('2d');

let camera = null;
let running = false;
let fpsCounter = {last: performance.now(), frames:0};

// Simple in-memory dataset: { label: [vectors...] }
const dataset = {};
let totalExamples = 0;

function updateDatasetUI(){ el('datasetCount').innerText = totalExamples; }

// Normalize landmarks to canonical vector (relative positions, scale-invariant)
function landmarksToVector(landmarks){
  // landmarks: array of {x,y,z} (normalized 0..1)
  // We'll translate so wrist (landmark 0) is origin and scale by maximum distance
  const pts = landmarks.map(p => [p.x, p.y, p.z || 0]);
  const origin = pts[0];
  const rel = pts.map(p => [p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]]);
  // compute scale
  let maxd = 0;
  for (let i=0;i<rel.length;i++){
    const d = Math.hypot(rel[i][0], rel[i][1], rel[i][2]);
    if (d>maxd) maxd=d;
  }
  const scale = maxd || 1;
  // flatten vector (2D or 3D)
  const flat = [];
  for (let i=0;i<rel.length;i++){
    flat.push(rel[i][0]/scale, rel[i][1]/scale, rel[i][2]/scale);
  }
  return flat; // length = 21*3 = 63
}

function euclidean(a,b){
  let s=0;
  for(let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d; }
  return Math.sqrt(s);
}

// k-NN classifier
function knnPredict(vector, k=5){
  const entries = [];
  for (const label in dataset){
    for (const v of dataset[label]){
      const d = euclidean(vector, v);
      entries.push({label, d});
    }
  }
  if(entries.length===0) return null;
  entries.sort((a,b)=>a.d-b.d);
  const top = entries.slice(0,k);
  // vote
  const counts = {};
  for (const t of top){ counts[t.label] = (counts[t.label]||0)+1; }
  let best=null, bestc=0;
  for (const l in counts){ if(counts[l]>bestc){ best=l; bestc=counts[l]; } }
  // confidence = proportion in top-k
  const confidence = bestc / Math.min(k, entries.length);
  return {label:best, confidence};
}

// Initialize MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});

hands.onResults(onResults);

// Camera
async function startCamera(){
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:640, height:480}});
  video.srcObject = stream;
  await video.play();
  camera = new Camera(video, { onFrame: async ()=> { await hands.send({image: video}); }, width:640, height:480 });
  camera.start();
  el('status').innerText = 'Status: camera running';
}

// Handle results
let lastPrediction = null;
async function onResults(results){
  // fps
  fpsCounter.frames++;
  const now = performance.now();
  if(now - fpsCounter.last > 1000){ el('fps').innerText = fpsCounter.frames; fpsCounter.frames=0; fpsCounter.last=now; }

  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(results.multiHandLandmarks && results.multiHandLandmarks.length>0){
    const lm = results.multiHandLandmarks[0];
    // draw simple
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    for(let i=0;i<lm.length;i++){
      const x = lm[i].x * canvas.width;
      const y = lm[i].y * canvas.height;
      ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fillStyle='rgba(0,212,255,0.9)'; ctx.fill(); ctx.stroke();
    }

    // classification if running
    if(running){
      const vec = landmarksToVector(lm);
      const pred = knnPredict(vec, 5);
      if(pred){
        // smoothing: require confidence threshold
        if(pred.confidence >= 0.5){
          lastPrediction = pred;
          el('recognized').textContent = pred.label + '  (conf: ' + (pred.confidence*100).toFixed(0) + '%)';
        } else {
          // low confidence
          el('recognized').textContent = '—';
        }
      }
    }
  } else {
    // no hand
  }
}

// UI actions
el('addExampleBtn').addEventListener('click', ()=>{
  const lbl = el('labelInput').value.trim();
  if(!lbl){ alert('Masukkan label/kata dahulu'); return; }
  // capture current frame landmarks if present
  // we can attempt to obtain last results by sending a synthetic capture: easiest: use hands.send with current video frame and listen in onResults
  hands.send({image: video}).then(()=> {
    // onResults will run & we need latest landmarks; but we don't have direct return here.
    // Instead, we access the last drawn landmarks by asking video frame via hands API not exposed — workaround: use a one-time handler
    // Simpler approach: we'll grab next onResults call's landmarks by setting a temp flag.
  }).catch(e=> console.warn(e));
  // To ensure capture, use a short timeout and collect last known landmarks stored globally.
  setTimeout(()=>{
    // We will store last landmarks each onResults call into window._lastLandmarks
    if(window._lastLandmarks){
      const vec = landmarksToVector(window._lastLandmarks);
      if(!dataset[lbl]) dataset[lbl] = [];
      dataset[lbl].push(vec);
      totalExamples++;
      updateDatasetUI();
      el('recognized').textContent = `Added example for "${lbl}" (total ${dataset[lbl].length})`;
    } else {
      alert('Tidak terdeteksi tangan. Pastikan kamera aktif dan tangan terlihat, lalu coba lagi.');
    }
  }, 250);
});

// store last landmarks in onResults
hands.onResults((r) => {
  // keep default onResults behavior above, so we call original handler
});
// We'll reattach correct handler to capture landmarks and then call our onResults:
hands.onResults((results) => {
  if(results.multiHandLandmarks && results.multiHandLandmarks.length>0){
    window._lastLandmarks = results.multiHandLandmarks[0];
  } else {
    window._lastLandmarks = null;
  }
  // call the main onResults logic to draw and classify:
  // (duplicate drawing/classify code here to ensure it runs)
  // clear
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(results.multiHandLandmarks && results.multiHandLandmarks.length>0){
    const lm = results.multiHandLandmarks[0];
    ctx.strokeStyle = 'rgba(255,255,255,0.6)';
    ctx.lineWidth = 2;
    for(let i=0;i<lm.length;i++){
      const x = lm[i].x * canvas.width;
      const y = lm[i].y * canvas.height;
      ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fillStyle='rgba(0,212,255,0.9)'; ctx.fill(); ctx.stroke();
    }
    if(running){
      const vec = landmarksToVector(lm);
      const pred = knnPredict(vec, 5);
      if(pred){
        if(pred.confidence >= 0.5){
          lastPrediction = pred;
          el('recognized').textContent = pred.label + '  (conf: ' + (pred.confidence*100).toFixed(0) + '%)';
        } else {
          el('recognized').textContent = '—';
        }
      }
    }
  }
  // fps count
  fpsCounter.frames++;
  const now = performance.now();
  if(now - fpsCounter.last > 1000){ el('fps').innerText = fpsCounter.frames; fpsCounter.frames=0; fpsCounter.last=now; }
});

// Predict controls
el('startPredictBtn').addEventListener('click', ()=>{
  if(totalExamples===0){ alert('Belum ada contoh. Tambahkan contoh untuk label terlebih dahulu.'); return; }
  running = true;
  el('status').innerText = 'Status: predicting';
});
el('stopPredictBtn').addEventListener('click', ()=>{
  running = false;
  el('status').innerText = 'Status: camera running';
});
el('clearBtn').addEventListener('click', ()=>{
  if(confirm('Hapus semua dataset?')){ for(const k in dataset) delete dataset[k]; totalExamples=0; updateDatasetUI(); el('recognized').textContent='—'; }
});

// export/import dataset
el('exportBtn').addEventListener('click', ()=>{
  const payload = JSON.stringify(dataset);
  const blob = new Blob([payload], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'gesture-dataset.json';
  a.click();
});
el('importBtn').addEventListener('click', ()=> el('importFile').click());
el('importFile').addEventListener('change', (ev)=>{
  const f = ev.target.files[0];
  if(!f) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try{
      const obj = JSON.parse(e.target.result);
      // merge
      let count=0;
      for(const k in obj){
        if(!dataset[k]) dataset[k]=[];
        for(const v of obj[k]){ dataset[k].push(v); count++; totalExamples++; }
      }
      updateDatasetUI();
      alert('Imported ' + count + ' examples');
    }catch(err){ alert('Invalid file'); }
  };
  reader.readAsText(f);
});

// speak & copy
el('speakBtn').addEventListener('click', async ()=>{
  let text = el('recognized').textContent || '';
  text = text.split('  (conf:')[0].trim();
  if(!text || text==='—'){ alert('Belum ada hasil untuk dibacakan'); return; }
  // optional translate
  const url = el('translateUrl').value.trim();
  if(url){
    try{
      const tgt = el('targetLang').value;
      const resp = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({q:text, source:'auto', target:tgt, format:'text'})});
      if(resp.ok){ const j=await resp.json(); // LibreTranslate returns {translatedText}
        text = j.translatedText || (j.result || JSON.stringify(j));
      } else {
        console.warn('translate failed', resp.status);
      }
    }catch(e){ console.warn('translate error', e); }
  }
  // speak
  const synth = window.speechSynthesis;
  const ut = new SpeechSynthesisUtterance(text);
  // choose voice optionally
  synth.cancel();
  synth.speak(ut);
});

// copy recognized
el('copyBtn').addEventListener('click', ()=> {
  const t = el('recognized').textContent.split('  (conf:')[0].trim();
  if(t && t!=='—'){ navigator.clipboard.writeText(t).then(()=> alert('Copied: ' + t)); } else alert('Nothing to copy');
});

// Start camera on load
(async ()=>{
  try{
    await startCamera();
    // ensure canvas size matches video
    video.addEventListener('loadedmetadata', ()=> {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    });
    // init fps
    fpsCounter.last = performance.now();
  }catch(e){
    alert('Gagal mengakses kamera: ' + e.message);
  }
})();
