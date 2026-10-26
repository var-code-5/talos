const $ = id => document.getElementById(id);
let state = null, selectedJob = null, page = 0, tasksShown = [];
let isolationChosen = false;
$('isolation').addEventListener('change', () => {isolationChosen = true;});
const pageSize = 8;
const escapeHTML = text => String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const label = text => String(text).replaceAll('_', ' ');
async function api(route, data) {
  const response = await fetch(route, data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
$('audit').addEventListener('change', () => {
  $('audit-help').textContent = $('audit').value === 'full' ? 'Recompute every remote gradient on the requester.' : '60% for new workers; 15% after five valid audits. Failures raise audits to 100%. Unsampled corruption can pass.';
});
$('run-form').addEventListener('submit', async event => {
  event.preventDefault(); $('form-error').textContent = ''; $('start').disabled = true;
  try {
    await api('/api/jobs', {steps:Number($('steps').value), audit:$('audit').value, scenario:$('scenario').value, isolation:$('isolation').value});
    selectedJob = null; page = 0; await refresh();
  } catch (error) { $('form-error').textContent = error.message; $('start').disabled = false; }
});
$('stop').addEventListener('click', async () => {
  try { await api('/api/cancel', {}); await refresh(); }
  catch(error) { $('form-error').textContent = error.message; }
});
$('close-detail').addEventListener('click', () => $('task-detail').close());
$('live-view').addEventListener('click', () => {selectedJob = null; page = 0; render();});
$('task-detail').addEventListener('click', event => { if(event.target === $('task-detail')) $('task-detail').close(); });
$('previous-page').addEventListener('click', () => {page--; renderTasks();});
$('next-page').addEventListener('click', () => {page++; renderTasks();});
$('task-rows').addEventListener('click', event => {
  const button = event.target.closest('[data-task]');
  if (!button) return;
  $('detail-json').textContent = JSON.stringify(tasksShown[Number(button.dataset.task)], null, 2);
  $('task-detail').showModal();
});
$('run-list').addEventListener('click', async event => {
  const button = event.target.closest('[data-run]');
  if (!button) return;
  try {
    const response = await api('/api/export');
    // Export includes full historical snapshots for inspection and reproducibility.
    selectedJob = response.jobs.find(job => job.id === button.dataset.run) || null;
    page = 0; render();
  } catch(error) { $('form-error').textContent = error.message; }
});
function chart(job) {
  const history = job.history;
  $('chart-empty').classList.add('hidden'); $('loss-chart').classList.remove('hidden');
  const width = 700, height = 195, left = 45, right = 18, top = 14, bottom = 30;
  const maximum = Math.max(history[0].loss, .01);
  const x = i => left + i / job.steps * (width - left - right);
  const y = value => top + (1 - value / maximum) * (height - top - bottom);
  let svg = '';
  for(let i = 0; i <= 3; i++) {
    const value = maximum * i / 3, py = y(value);
    svg += `<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}" stroke="#e7edf4"/><text x="${left-9}" y="${py+4}" text-anchor="end" fill="#8a9bad" font-size="10">${value.toFixed(2)}</text>`;
  }
  for (const step of [0, Math.round(job.steps/2), job.steps]) svg += `<text x="${x(step)}" y="${height-9}" text-anchor="middle" fill="#8a9bad" font-size="10">${step}</text>`;
  const line = history.map(p => `${x(p.step)},${y(p.loss)}`).join(' ');
  const last = history.at(-1);
  svg += `<polygon points="${left},${y(0)} ${line} ${x(last.step)},${y(0)}" fill="#edf6f3"/><polyline points="${line}" fill="none" stroke="#258774" stroke-width="2.3" stroke-linejoin="round"/><circle cx="${x(last.step)}" cy="${y(last.loss)}" r="4" fill="#258774" stroke="white" stroke-width="2"/>`;
  $('loss-chart').innerHTML = svg;
  $('loss-chart').setAttribute('aria-label', `Training loss decreased from ${history[0].loss.toFixed(4)} to ${last.loss.toFixed(4)} over ${last.step} steps.`);
}
function renderTasks() {
  const job = selectedJob || state?.job;
  if (!job) return;
  tasksShown = [...job.tasks].reverse();
  const pages = Math.max(1, Math.ceil(tasksShown.length/pageSize));
  page = Math.max(0, Math.min(page,pages-1));
  $('task-pagination').classList.toggle('hidden', pages <= 1);
  $('page-label').textContent = `${page+1} / ${pages} · ${tasksShown.length} work units`;
  $('previous-page').disabled = page === 0; $('next-page').disabled = page >= pages-1;
  $('task-rows').innerHTML = tasksShown.slice(page*pageSize, (page+1)*pageSize).map((task, i) => {
    const c = task.contract, recovered = Boolean(task.recovery);
    const audit = task.audit === true ? 'Independent audit' : task.audit === false ? 'Accepted without audit' : '—';
    return `<tr><td>${escapeHTML(c.task_id)} <small> / a${c.attempt}</small></td><td>${escapeHTML(task.worker_id)}</td><td>v${c.model_version}</td><td>${c.payload.end-c.payload.start}</td><td><span class="badge ${escapeHTML(task.status)}">${escapeHTML(label(task.status))}</span>${recovered ? ' ↻ recovered' : ''}</td><td>${audit}</td><td><button class="detail-button" data-task="${page*pageSize+i}">Inspect</button></td></tr>`;
  }).join('') || '<tr><td colspan="7" class="empty-cell">Creating first assignments…</td></tr>';
}
function render() {
  const job = selectedJob || state.job;
  $('live-view').classList.toggle('hidden', !selectedJob);
  const running = state.job?.status === 'running';
  const online = state.workers.filter(w => w.online).length;
  $('connection').textContent = `${online} volunteers connected`;
  $('connection').classList.remove('disconnected');
  $('start').disabled = running || online < 1;
  $('start').innerHTML = running ? '<span>◌</span> Training in progress' : '<span>▷</span> Start training';
  $('stop').classList.toggle('hidden', !running);
  for (const id of ['scenario','audit','steps','isolation']) $(id).disabled = running;
  const workers = [...state.workers].sort((a,b) => a.id.localeCompare(b.id));
  if (!isolationChosen && online > 0) {
    $('isolation').value = workers.filter(w => w.online).every(w => w.policy.isolation === 'kvm-microvm') ? 'kvm-microvm' : 'bounded-process';
    isolationChosen = true;
  }
  $('cluster-total').textContent = `Requester + ${online} online volunteers`;
  const capacityRows = [{id:'requester', endpoint:'Local coordinator', resources:state.requester_resources, online:true}, ...workers];
  $('resource-rows').innerHTML = capacityRows.map(worker => {
    const r = worker.resources || {}, c = worker.connection;
    return `<tr><td><strong>${escapeHTML(worker.id)}</strong><br><small>${escapeHTML(worker.endpoint || 'Legacy pull worker')}</small></td><td>${r.offered_vcpus || '—'} vCPU<br><small>${r.gradient_samples_per_second ? Number(r.gradient_samples_per_second).toLocaleString() : '—'} samples/s host kernel</small></td><td>${r.offered_memory_mb || '—'} MiB</td><td>${r.offered_upload_mbps || '—'} / ${r.offered_download_mbps || '—'} Mbps</td><td>${c ? `${c.node_upload_mbps} / ${c.node_download_mbps} Mbps` : 'Local / not probed'}</td><td><span class="badge ${worker.online ? 'accepted' : 'timed_out'}">${worker.online ? worker.status === 'computing' ? 'Computing' : 'Accepting' : 'Offline'}</span><br><small>${c ? c.rtt_ms + ' ms round trip' : ''}</small></td></tr>`;
  }).join('');
  const partitions = job?.partitions || [];
  const participant = (name, role, count, status, description, local=false) => `<article class="node ${local ? 'requester' : ''}"><div class="node-head"><span class="device">▣</span><span class="node-role">${escapeHTML(role)}</span><span class="status-dot ${status === 'Offline' ? 'offline' : ''}"></span></div><h3>${escapeHTML(name)}</h3><p>${escapeHTML(description)}</p><div class="node-bottom"><strong>${count}</strong><span>samples / step</span></div></article>`;
  document.querySelector('.nodes').innerHTML = participant('Your machine','Requester',400,'Connected','Train + verify + aggregate',true) + workers.map(worker => {
    const index = job?.workers.indexOf(worker.id) ?? -1;
    const partition = index >= 0 ? partitions[index] : null;
    return participant(worker.id, worker.online ? 'Volunteer' : 'Offline', partition ? partition[1]-partition[0] : '—', worker.online ? 'Connected' : 'Offline', worker.policy.isolation === 'kvm-microvm' ? 'Firecracker microVM' : 'Bounded process');
  }).join('');
  $('run-total').textContent = `${state.runs.length} runs`;
  $('run-list').innerHTML = state.runs.map(run => `<div class="run-item"><div><button class="run-select" data-run="${escapeHTML(run.id)}">${escapeHTML(label(run.scenario))} · ${run.steps} steps</button><p>${new Date(run.created_at*1000).toLocaleTimeString()} · ${escapeHTML(run.audit)} audits</p></div><span class="tag ${escapeHTML(run.status)}">${escapeHTML(run.status)}</span></div>`).join('') || '<p class="empty-log">Completed runs are saved locally in SQLite.</p>';
  if (!job) return;
  $('run-status').textContent = selectedJob ? `${job.status} · historical run` : job.status;
  $('run-status').className = `tag ${job.status}`;
  const m = job.metrics, history = job.history, latest = history.at(-1);
  $('model-version').textContent = `v${latest.step}`;
  $('step-count').textContent = `${latest.step} of ${job.steps} steps`;
  $('accepted-count').textContent = m.accepted;
  $('audit-count').textContent = `${m.audited} audited · ${m.unaudited} unaudited`;
  $('rejected-count').textContent = `${m.rejected + m.timeouts} / ${m.recovered}`;
  $('rejection-note').textContent = m.policy_rejections ? `${m.policy_rejections} policy rejections` : `${m.timeouts} timeouts · trusted recovery`;
  $('loss-value').textContent = latest.loss.toFixed(5);
  $('loss-change').textContent = `${((1-latest.loss/history[0].loss)*100).toFixed(1)}% below initial loss`;
  $('weights').textContent = `Learned weights: ${job.weights.map(w => w.toFixed(3)).join(', ')}`;
  chart(job); renderTasks();
  $('event-list').innerHTML = [...job.events].reverse().slice(0,80).map(event => `<div class="event"><span class="event-icon ${escapeHTML(event.kind)}">${['rejected','timeout','error'].includes(event.kind) ? '!' : '✓'}</span><p>${escapeHTML(event.message)}</p><time>${new Date(event.time*1000).toLocaleTimeString()}</time></div>`).join('');
}
let refreshing = false;
async function refresh() {
  if(refreshing) return;
  refreshing = true;
  try {state = await api('/api/state'); render();}
  catch(error) {$('connection').textContent = 'Coordinator unavailable'; $('connection').classList.add('disconnected'); $('start').disabled = true;}
  finally {refreshing = false;}
}
refresh(); setInterval(refresh,700);
