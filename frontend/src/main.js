const fileInput=document.querySelector('#fileInput'),dropZone=document.querySelector('#dropZone'),fileCard=document.querySelector('#fileCard'),processButton=document.querySelector('#processButton'),preview=document.querySelector('#localPreview'),resultVideo=document.querySelector('#resultVideo'),progress=document.querySelector('#progress');let objectUrl,selectedFile;
const srtList=document.querySelector('#srtList'),reportList=document.querySelector('#reportList');
const formatTime=seconds=>{const value=Math.max(0,Math.floor(seconds));return`${String(Math.floor(value/60)).padStart(2,'0')}:${String(value%60).padStart(2,'0')}`};
function setProgress(value,label){progress.classList.remove('hidden');progress.querySelector('.progress-track i').style.width=`${value}%`;document.querySelector('#progressPercent').textContent=`${Math.round(value)}%`;document.querySelector('#progressLabel').textContent=label;const active=value<10?0:value<70?1:value<96?2:3;document.querySelectorAll('.stage').forEach((stage,index)=>stage.classList.toggle('active',index<=active))}
function selectFile(file){if(!file||!file.type.startsWith('video/'))return;selectedFile=file;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(file);preview.src=objectUrl;document.querySelector('#fileName').textContent=file.name;document.querySelector('#fileMeta').textContent=`${(file.size/1024/1024).toFixed(1)} MB · Ready to process`;dropZone.classList.add('hidden');fileCard.classList.remove('hidden');processButton.classList.remove('hidden');processButton.disabled=false;progress.classList.add('hidden');document.querySelector('#lookupError').classList.add('hidden');document.querySelector('#results').classList.add('hidden')}

// ---- Single shared state driving both columns. is_guess now means what
// it originally meant: this moment is a blank the model wasn't confident
// enough to fill in on its own. Confident captions (is_guess:false) are
// locked/read-only in the report; only blanks are editable. Filling in a
// blank flips it to is_guess:false, both locally and on the server, so it
// becomes real text everywhere and stops being editable. Scores/candidate
// data still travels with each item in state, computed and kept
// server-side — deliberately never rendered here. ----
const state={jobId:null,timeline:[]};

function renderSrtList(items){
  srtList.innerHTML=items.map(item=>{
    const blank=item.is_guess;
    return`<div class="srt-row${blank?' is-blank':''}" data-id="${item.id}" data-search="${item.caption} ${(item.families||[]).join(' ')}"><img src="${item.frame_url}" alt="" loading="lazy"/><div><time>${formatTime(item.start_sec)} — ${formatTime(item.end_sec)}</time><div class="srt-caption">${blank?'<em>— blank, not yet filled in —</em>':item.caption}</div></div></div>`
  }).join('');
}

function renderReportList(items){
  reportList.innerHTML=items.map(item=>{
    if(item.is_guess){
      // A blank: always shown as an editable field, since it inherently
      // needs input — no separate "click to edit" step for these.
      return`<article class="report-row is-blank glass" data-id="${item.id}" data-search="${item.caption} ${(item.families||[]).join(' ')}"><time>${formatTime(item.start_sec)} — ${formatTime(item.end_sec)}</time><div class="blank-label">Blank — model wasn't confident here</div><textarea rows="2" placeholder="Type the correct caption…" data-blank-input="${item.id}"></textarea><div class="report-row-actions"><button class="row-action-button save-blank" data-save-blank="${item.id}" type="button">Save &amp; next blank →</button><button class="row-action-button delete-btn" data-delete="${item.id}" type="button">🗑 Not needed</button></div></article>`;
    }
    // A confident, final caption: read-only, no edit control — deletable
    // only, since editing final captions isn't allowed from here.
    return`<article class="report-row glass" data-id="${item.id}" data-search="${item.caption} ${(item.families||[]).join(' ')}"><time>${formatTime(item.start_sec)} — ${formatTime(item.end_sec)}</time><div class="report-caption-row"><h3>${item.caption}</h3><div class="report-row-actions"><button class="row-action-button delete-btn" data-delete="${item.id}" type="button">🗑 Delete</button></div></div></article>`;
  }).join('');
  updateFinalizeBar();
}

function renderBoth(){
  renderSrtList(state.timeline);
  renderReportList(state.timeline);
}

function updateFinalizeBar(){
  const remaining=state.timeline.filter(t=>t.is_guess).length;
  const bar=document.querySelector('#finalizeBar');
  document.querySelector('#blanksRemaining').textContent=remaining;
  bar.classList.toggle('hidden',state.timeline.length===0);
  document.querySelector('#finalizeBtn').disabled=false;
}

async function patchCaption(captionId,newText){
  const response=await fetch(`/api/jobs/${state.jobId}/captions/${captionId}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({caption:newText})});
  if(!response.ok){let detail='Could not save this caption.';try{detail=(await response.json()).detail||detail}catch{}throw new Error(detail)}
  return response.json()
}

async function deleteCaptionRequest(captionId){
  const response=await fetch(`/api/jobs/${state.jobId}/captions/${captionId}`,{method:'DELETE'});
  if(!response.ok){let detail='Could not remove this caption.';try{detail=(await response.json()).detail||detail}catch{}throw new Error(detail)}
  return response.json()
}

function focusNextBlank(afterId){
  const blanks=state.timeline.filter(t=>t.is_guess);
  if(!blanks.length)return;
  const next=blanks.find(t=>t.id>afterId)||blanks[0];
  const row=document.querySelector(`.report-row[data-id="${next.id}"]`);
  if(row){row.scrollIntoView({behavior:'smooth',block:'center'});const box=row.querySelector('textarea');if(box)box.focus()}
}

reportList.addEventListener('click',async e=>{
  const saveBlank=e.target.closest('[data-save-blank]');
  if(saveBlank){
    const id=Number(saveBlank.dataset.saveBlank);
    const box=document.querySelector(`[data-blank-input="${id}"]`);
    const newText=box.value.trim();
    if(!newText){box.focus();return}
    saveBlank.disabled=true;saveBlank.textContent='Saving…';
    try{
      const result=await patchCaption(id,newText);
      const item=state.timeline.find(t=>t.id===id);
      if(item){item.caption=result.caption;item.is_guess=false}
      renderBoth();
      focusNextBlank(id);
    }catch(error){
      saveBlank.disabled=false;saveBlank.textContent='Save & next blank →';
      alert(error.message);
    }
    return;
  }
  const del=e.target.closest('[data-delete]');
  if(del){
    const id=Number(del.dataset.delete);
    if(!confirm('Remove this caption entirely? This updates the SRT file immediately.'))return;
    del.disabled=true;
    try{
      await deleteCaptionRequest(id);
      state.timeline=state.timeline.filter(t=>t.id!==id);
      renderBoth();
    }catch(error){
      del.disabled=false;
      alert(error.message);
    }
  }
});

// ---- Finalize: any blank nobody filled in gets dropped entirely from
// the SRT and the report, rather than shipping as an empty gap. ----
document.querySelector('#finalizeBtn').addEventListener('click',async()=>{
  const remaining=state.timeline.filter(t=>t.is_guess);
  const button=document.querySelector('#finalizeBtn');
  if(remaining.length&&!confirm(`${remaining.length} blank(s) were never filled in. Finalizing will remove them from the captions entirely. Continue?`))return;
  button.disabled=true;button.textContent='Finalizing…';
  try{
    for(const item of remaining){
      await deleteCaptionRequest(item.id);
    }
    state.timeline=state.timeline.filter(t=>!t.is_guess);
    renderBoth();
    button.textContent='✓ Finalized';
  }catch(error){
    alert(error.message);
    button.disabled=false;button.textContent='Save all & finalize';
  }
});

function createJob(file){return new Promise((resolve,reject)=>{const request=new XMLHttpRequest(),form=new FormData();form.append('video',file);request.open('POST','/api/jobs');request.upload.addEventListener('progress',event=>{if(event.lengthComputable)setProgress(Math.max(2,event.loaded/event.total*8),'Uploading video…')});request.addEventListener('load',()=>{let body={};try{body=JSON.parse(request.responseText)}catch{}if(request.status>=200&&request.status<300)resolve(body);else reject(new Error(body.detail||'The video could not be uploaded.'))});request.addEventListener('error',()=>reject(new Error('Cannot reach the processing server. Make sure the backend is running.')));request.send(form)})}
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));async function waitForJob(url){while(true){const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw new Error('Could not read processing status.');const job=await response.json();setProgress(job.progress||10,job.stage||'Processing video…');if(job.status==='completed')return job;if(job.status==='failed')throw new Error(job.error||'The ML pipeline failed.');await wait(3000)}}
async function loadResult(url){
  const response=await fetch(url);
  if(!response.ok)throw new Error('The result was created but could not be loaded.');
  const data=await response.json();
  state.jobId=data.job_id;state.timeline=data.timeline;
  resultVideo.src=data.video_url;resultVideo.muted=false;resultVideo.volume=1;resultVideo.load();
  document.querySelector('#summaryCount').textContent=`${data.caption_segments} non-speech moments`;
  document.querySelector('#statCount').textContent=data.caption_segments;
  document.querySelector('#statDuration').textContent=formatTime(data.duration_sec);
  document.querySelector('#srtDownload').href=data.srt_url;
  document.querySelector('#videoDownload').href=data.video_url;
  renderBoth();
}
document.querySelector('#browseButton').addEventListener('click',e=>{e.stopPropagation();fileInput.click()});dropZone.addEventListener('click',()=>fileInput.click());dropZone.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')fileInput.click()});fileInput.addEventListener('change',()=>selectFile(fileInput.files[0]));['dragenter','dragover'].forEach(n=>dropZone.addEventListener(n,e=>{e.preventDefault();dropZone.classList.add('dragging')}));['dragleave','drop'].forEach(n=>dropZone.addEventListener(n,e=>{e.preventDefault();dropZone.classList.remove('dragging')}));dropZone.addEventListener('drop',e=>selectFile(e.dataTransfer.files[0]));document.querySelector('#removeFile').addEventListener('click',()=>{selectedFile=null;fileInput.value='';fileCard.classList.add('hidden');dropZone.classList.remove('hidden');processButton.disabled=true;progress.classList.add('hidden');document.querySelector('#results').classList.add('hidden')});
processButton.addEventListener('click',async()=>{processButton.classList.add('hidden');const errorBox=document.querySelector('#lookupError');errorBox.classList.add('hidden');try{setProgress(2,'Uploading video…');const created=await createJob(selectedFile);setProgress(9,'Video uploaded. Starting pipeline…');const completed=await waitForJob(created.status_url);await loadResult(completed.result_url);progress.classList.add('hidden');document.querySelector('#results').classList.remove('hidden');document.querySelector('#results').scrollIntoView({behavior:'smooth'})}catch(error){errorBox.textContent=error.message;errorBox.classList.remove('hidden');progress.classList.add('hidden');processButton.classList.remove('hidden')}});
document.querySelector('#searchInput').addEventListener('input',e=>{
  const q=e.target.value.toLowerCase();
  document.querySelectorAll('.srt-row,.report-row').forEach(row=>row.classList.toggle('filtered',!row.dataset.search.toLowerCase().includes(q)&&!row.textContent.toLowerCase().includes(q)))
});
const soundToggle=document.querySelector('#soundToggle');function updateSoundLabel(){const enabled=!resultVideo.muted&&resultVideo.volume>0;soundToggle.textContent=enabled?'◖)) Sound on':'◖× Sound off';soundToggle.classList.toggle('is-muted',!enabled)}soundToggle.addEventListener('click',()=>{resultVideo.muted=!resultVideo.muted;if(!resultVideo.muted&&resultVideo.volume===0)resultVideo.volume=1;updateSoundLabel()});resultVideo.addEventListener('volumechange',updateSoundLabel);
