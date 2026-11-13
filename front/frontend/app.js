const $ = (q, root = document) => root.querySelector(q);
const $$ = (q, root = document) => [...root.querySelectorAll(q)];
const state = { meta: null, data: null, scenarios: [], demo: null, currentCycle: null, selectedPid: null, busy: false, mode: 'scenario', menuResource: null };
const escapeHtml = (value = '') => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const titleCase = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const api = async (path, options = {}) => {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
};
const setBusy = value => { state.busy = value; document.body.classList.toggle('busy', value); $$('button,select,input').forEach(el => { if (!el.dataset.keepEnabled) el.disabled = value; }); };
async function request(path, body, errorTarget) {
  if (state.busy) return null;
  setBusy(true); if (errorTarget) $(errorTarget).textContent = '';
  try { const result = await api(path, { method: 'POST', body: JSON.stringify(body || {}) });
    if (result.state) { state.data = result.state; state.currentCycle = result.cycle || null; render(); }
    if (/\/api\/(demo\/next|event|patients|resources\/|patients\/.*\/(vitals|discharge))/.test(path)) highlightChanges();
    return result;
  } catch (error) { if (errorTarget) $(errorTarget).textContent = error.message; else toast(error.message); return null; }
  finally { setBusy(false); }
}
function toast(text) { const el = $('#toast'); el.textContent = text; el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 2500); }
function highlightChanges() { if (matchMedia('(prefers-reduced-motion: reduce)').matches) return; $$('.patient-row, .bed').forEach(el => { el.classList.remove('pulse'); void el.offsetWidth; el.classList.add('pulse'); }); setTimeout(() => $$('.pulse').forEach(el => el.classList.remove('pulse')), 2600); }
function riskClass(risk) { return `risk-${String(risk || '').toLowerCase()}`; }
function riskDot(risk) { return `<span class="risk-dot ${escapeHtml(String(risk || '').toLowerCase())}" aria-hidden="true"></span>`; }
function patientStatus(p) {
  if ((state.data?.allocation?.waiting || []).includes(p.pid)) return 'Left waiting';
  if (p.status === 'waiting') return 'Waiting';
  return p.holds?.length ? 'Admitted' : 'Reserved';
}
function groupBeds(resources) {
  const groups = [{ label: 'ICU', match: r => /icu/i.test(`${r.type || ''} ${r.id}`) }, { label: 'Emergency', match: r => /emergency|eb|resus/i.test(`${r.type || ''} ${r.id}`) }, { label: 'Exam room', match: r => /exam|room/i.test(`${r.type || ''} ${r.id}`) }];
  const remaining = [...resources]; const out = [];
  for (const group of groups) { const beds = remaining.filter(group.match); beds.forEach(b => remaining.splice(remaining.indexOf(b), 1)); if (beds.length) out.push({ label: group.label, beds }); }
  if (remaining.length) out.push({ label: 'Other beds', beds: remaining }); return out;
}
function renderBeds() {
  const resources = state.data?.resources || []; const patients = state.data?.patients || [];
  const beds = resources.filter(r => r.kind === 'bed'); $('#bed-count').textContent = `${beds.length} beds`;
  $('#bed-groups').innerHTML = groupBeds(beds).map(group => `<div class="bed-group"><h3>${escapeHtml(group.label)}</h3><div class="bed-grid">${group.beds.map(b => {
    const p = patients.find(x => x.pid === b.held_by); const risk = p?.risk || ''; const status = b.status === 'out_of_service' ? 'out-of-service' : riskClass(risk);
    const statusText = b.status === 'out_of_service' ? 'Out of service' : p ? `${escapeHtml(p.pid)} · ${patientStatus(p) === 'Reserved' ? 'reserved' : 'admitted'}` : 'Available';
    const equipment = p?.holds?.filter(id => resources.find(r => r.id === id)?.kind === 'equipment') || [];
    return `<button class="bed ${status}" data-resource="${escapeHtml(b.id)}" aria-label="${escapeHtml(b.id)} ${escapeHtml(statusText)}"><span class="risk-stripe"></span><span class="bed-id">${escapeHtml(b.id)}</span><div class="bed-patient">${escapeHtml(statusText)}</div>${equipment.length ? `<div class="bed-equipment">${equipment.map(escapeHtml).join(' · ')}</div>` : ''}</button>`;
  }).join('')}</div></div>`).join('') || '<div class="empty-state">No beds listed.</div>';
  const staff = resources.filter(r => ['doctor','nurse'].includes(String(r.kind).toLowerCase()));
  const equipment = resources.filter(r => r.kind === 'equipment');
  const chip = r => `<button class="resource-chip status-${escapeHtml(r.status)}" data-resource="${escapeHtml(r.id)}" aria-label="${escapeHtml(r.id)} ${escapeHtml(r.type_label || r.type || r.kind)} ${escapeHtml(r.status)}"><strong>${escapeHtml(r.id)}</strong> ${escapeHtml(r.type_label || r.type || r.kind)} <span class="resource-owner">· ${escapeHtml(r.held_by || titleCase(r.status))}</span></button>`;
  $('#staff-list').innerHTML = staff.map(chip).join('') || '<span class="muted-note">No staff listed.</span>';
  $('#equipment-list').innerHTML = equipment.map(chip).join('') || '<span class="muted-note">No equipment listed.</span>';
  $$('[data-resource]').forEach(el => el.addEventListener('click', event => showResourceMenu(event, el.dataset.resource)));
}
function sparkline(history, key) {
  const values = (history || []).map(row => Number(row[key])).filter(Number.isFinite); if (values.length < 2) return '';
  const min = Math.min(...values), max = Math.max(...values); const points = values.map((v, i) => `${(i / (values.length - 1)) * 38},${12 - (max === min ? 6 : ((v - min) / (max - min)) * 10)}`).join(' ');
  return `<svg class="sparkline" viewBox="0 0 40 14" role="img" aria-label="${escapeHtml(key)} history"><polyline points="${points}" fill="none" stroke="currentColor" stroke-width="1.4" vector-effect="non-scaling-stroke"/></svg>`;
}
// Same comparisons as the rules in src/reasoning/prolog/findings.pl
const VITAL_RULES = { hr: ['hr_high', 'above'], sbp: ['sbp_low', 'below'], map: ['map_low', 'below'], spo2: ['spo2_low', 'below'], temp: ['temp_fever', 'atleast'], rr: ['rr_high', 'atleast'], lactate: ['lactate_high', 'atleast'] };
const VITAL_SHORT = { hr: 'HR', sbp: 'SBP', dbp: 'DBP', map: 'MAP', spo2: 'SpO2', temp: 'T', rr: 'RR', lactate: 'Lact' };
function abnormalCheck(key, value) { const rule = VITAL_RULES[key]; const limit = rule && state.meta?.thresholds?.[rule[0]]; if (limit == null || value == null) return null; const v = Number(value);
  const bad = rule[1] === 'above' ? v > limit : rule[1] === 'below' ? v < limit : v >= limit; return bad ? `${rule[1] === 'below' ? 'Below' : 'Above'} threshold ${limit}` : null; }
function renderPatients() {
  const patients = state.data?.patients || []; $('#patient-count').textContent = `${patients.length} patients`; $('#queue-empty').classList.toggle('hidden', patients.length > 0);
  const vitalMeta = state.meta?.vitals || [];
  $('#patient-queue').innerHTML = patients.map(p => {
    const measurements = vitalMeta.map(v => { const value = p.vitals?.[v.key]; if (value == null) return ''; const abnormal = abnormalCheck(v.key, value); const unit = v.unit ? `<span class="unit"> ${escapeHtml(v.unit)}</span>` : '';
      return `<span class="vital-value ${abnormal ? 'abnormal' : ''}" ${abnormal ? `title="${escapeHtml(abnormal)} ${escapeHtml(v.unit || '')}"` : ''}>${escapeHtml(VITAL_SHORT[v.key] || v.label || v.key)} ${escapeHtml(value)}${unit}${abnormal ? '<span aria-label="abnormal"> !</span>' : ''}${sparkline(p.history, v.key)}</span>`;
    }).filter(Boolean).join('');
    const model = ['ml','dl'].filter(k => p.predictions?.[k]).map(k => { const pred = p.predictions[k]; const [level, prob] = pred; const low = state.meta?.thresholds?.model_confidence != null && prob < state.meta.thresholds.model_confidence;
      return `<div class="prediction-line ${low ? 'low-confidence' : ''}">${k.toUpperCase()} ${escapeHtml(level)} ${Math.round(prob * 100)}%${low ? '<span class="confidence-note"> · below confidence threshold</span>' : ''}</div>`;
    }).join('') || '<span class="prediction-line muted-note">No model estimate</span>';
    return `<article class="patient-row ${p._changed ? 'pulse' : ''}" tabindex="0" role="button" data-patient="${escapeHtml(p.pid)}" aria-label="Open ${escapeHtml(p.pid)} details"><div class="patient-main"><span class="patient-id">${escapeHtml(p.pid)}</span><div><div>${escapeHtml(p.age)} y${p.sex ? ` · ${escapeHtml(p.sex.toUpperCase())}` : ''}</div><div class="patient-demographics">${(p.symptoms || []).slice(0,2).map(titleCase).map(escapeHtml).join(', ')}</div></div></div><div><span class="risk-chip">${riskDot(p.risk)}${escapeHtml(p.risk || 'Unrated')}</span><span class="patient-status">${escapeHtml(patientStatus(p))}</span></div><div class="vitals-line">${measurements || '<span class="muted-note">No current observations</span>'}</div><div>${model}</div></article>`;
  }).join('');
  $$('.patient-row').forEach(row => { const open = () => openPatient(row.dataset.patient); row.addEventListener('click', open); row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } }); });
}
function renderTimeline() {
  const executed = state.data?.executed || [], plan = state.data?.plan || []; const items = [...executed.map((a, i) => ({a, cls:'done', label:`Executed ${i + 1}`})), ...plan.map((a, i) => ({a, cls:i === 0 ? 'next' : 'future', label:i === 0 ? 'Next action' : `Upcoming ${i}`}))];
  $('#timeline').innerHTML = items.map(({a,cls,label}, i) => `<div class="timeline-item ${cls}"><span class="timeline-num">${escapeHtml(label)}</span>${escapeHtml(a.label || a.name || JSON.stringify(a))}</div>`).join('') || '<div class="empty-state">No actions in the current plan.</div>';
}
function renderCycle(cycle) {
  const cycleData = cycle || state.currentCycle; $('#cycle-history').innerHTML = (state.data?.cycles || []).map(c => `<button class="cycle-button ${cycleData?.number === c.number ? 'active' : ''}" data-cycle="${c.number}">Cycle ${c.number}${c.valid ? '' : ' · invalid'}</button>`).join('') || '<span class="muted-note">No deliberations yet.</span>';
  $$('.cycle-button').forEach(button => button.addEventListener('click', async () => { try { state.currentCycle = await api(`/api/cycles/${button.dataset.cycle}`); renderAgent(state.currentCycle); } catch (e) { toast(e.message); } }));
  if (!cycleData) { $('#cycle-badge').textContent = 'No cycle'; $('#agent-content').innerHTML = '<div class="empty-state">Waiting for the first deliberation.</div>'; return; }
  $('#cycle-badge').textContent = `Cycle ${cycleData.number}`; renderAgent(cycleData);
}
function listText(value) { return Array.isArray(value) ? value.map(x => typeof x === 'string' ? x : JSON.stringify(x)).join(' · ') : typeof value === 'object' && value ? Object.entries(value).map(([k,v]) => `${k}: ${Array.isArray(v) ? v.join(', ') : v}`).join(' · ') : String(value ?? 'None'); }
function renderAgent(c) {
  const reasons = c.reasons || []; const invalid = c.invalid_step;
  const changes = c.changes || []; const predictions = c.new_predictions || {}; const requirements = c.requirements || {};
  const decision = c.search || {}; const allocation = c.allocation || {};
  const searchCard = (name, s) => s ? `<div class="search-card"><strong>${escapeHtml(name)}</strong><div class="search-metrics">${s.found ? 'Found' : 'Not found'} · cost ${escapeHtml(s.cost ?? '—')}<br>${escapeHtml(s.expanded ?? 0)} expanded · ${escapeHtml(s.ms ?? 0)} ms</div></div>` : `<div class="search-card"><strong>${escapeHtml(name)}</strong><span class="muted-note">Unavailable</span></div>`;
  const actions = c.plan?.steps || []; const requirementsLine = Object.entries(requirements.patients || {}).map(([pid, r]) => { const needs = [r.bed && `${r.bed} bed`, r.specialist || r.doctor, r.nurse && 'nurse', ...(r.equipment || [])].filter(Boolean).join(', ');
      const flags = [r.urgent_evaluation && 'urgent', r.reassess && 'reassess'].filter(Boolean).join(', ');
      return `<p><span class="mono">${escapeHtml(pid)}</span> ${riskDot(r.risk)}<strong>${escapeHtml(r.risk || 'Unrated')}</strong> · ${r.waiting ? 'waiting' : 'under treatment'} · needs ${escapeHtml(needs || 'nothing')}${flags ? ` · ${escapeHtml(flags)}` : ''}</p>`; }).join('');
  const changed = Array.isArray(changes) ? changes.map(x => `<div class="change-line">${escapeHtml(typeof x === 'string' ? x : listText(x))}</div>`).join('') : Object.entries(changes).map(([k,v]) => `<div class="change-line"><span class="mono">${escapeHtml(k)}</span> ${escapeHtml(listText(v))}</div>`).join('');
  const predicted = Object.entries(predictions).map(([pid, sources]) => `<p><span class="mono">${escapeHtml(pid)}</span> ${Object.entries(sources || {}).map(([source, value]) => `${escapeHtml(source.toUpperCase())} ${escapeHtml(Array.isArray(value) ? `${value[0]} ${Number(value[1]).toFixed(2)}` : listText(value))}`).join(' · ')}</p>`).join('') || '<p class="muted-note">No new predictions.</p>';
  const allocations = allocation.patients || allocation; const allocationText = Object.entries(allocations).filter(([k]) => k !== 'waiting').map(([pid, a]) => { const parts = a && typeof a === 'object' ? [a.bed && `bed ${a.bed}`, a.doctor && `doctor ${a.doctor}`, a.nurse && `nurse ${a.nurse}`, ...Object.values(a.equipment || {})].filter(Boolean).join(', ') : listText(a);
      return `<div class="change-line"><span class="mono">${escapeHtml(pid)}</span> ${escapeHtml(a?.risk ? `${a.risk}: ` : '')}${escapeHtml(parts)}</div>`; }).join('')
      + (allocation.relocations || []).map(m => `<div class="change-line">Move <span class="mono">${escapeHtml(m.patient)}</span> ${escapeHtml(m.from)} → ${escapeHtml(m.to)}</div>`).join('')
      + ((allocation.waiting || []).length ? `<div class="change-line">Waiting: ${escapeHtml(allocation.waiting.join(', '))}</div>` : '');
  const announcements = c.search?.astar?.announcements || c.announcements || [];
  const diagnosis = c.search?.astar?.diagnosis || c.diagnosis;
  $('#agent-content').innerHTML = `<section class="agent-section"><h3>Why the agent re-decided</h3>${reasons.length ? reasons.map(r => `<p>${escapeHtml(typeof r === 'string' ? r : listText(r))}</p>`).join('') : '<p class="muted-note">Initial deliberation.</p>'}${invalid ? `<div class="invalid-block"><span class="mono">${escapeHtml(invalid.action)}</span> can no longer run: ${escapeHtml(listText(invalid.missing))}</div>` : ''}${changed}</section><section class="agent-section"><h3>Learning</h3>${predicted}</section><section class="agent-section"><h3>Reasoning</h3>${requirementsLine || '<p class="muted-note">No requirements available.</p>'}</section><section class="agent-section"><h3>Search</h3><div class="search-grid">${searchCard('A*', decision.astar)}${searchCard('Uniform cost', decision.ucs)}</div>${!c.valid ? `<div class="invalid-block"><strong>No valid allocation</strong><p>${escapeHtml(listText(diagnosis || 'No diagnosis supplied.'))}</p></div>` : ''}${allocationText}<p>${escapeHtml(listText(c.allocation_changes || []))}</p>${(Array.isArray(announcements) ? announcements : [announcements]).filter(Boolean).map(a => `<div class="warning-block">! ${escapeHtml(typeof a === 'string' ? a : listText(a))}</div>`).join('')}</section><section class="agent-section"><h3>Planning</h3>${actions.map((a,i) => `<details class="plan-step"><summary>${i+1}. ${escapeHtml(a.label || a.name || JSON.stringify(a))}</summary><pre>Preconditions\n${escapeHtml(listText(a.pre))}\n\nEffects\n+ ${escapeHtml(listText(a.add))}\n− ${escapeHtml(listText(a.delete))}</pre></details>`).join('') || '<p class="muted-note">No plan steps.</p>'}<p>${c.plan?.errors?.length ? `Plan validation: ${escapeHtml(listText(c.plan.errors))}` : 'Plan checked: every precondition holds'}</p></section>`;
}
function renderDemo() { const events = state.demo?.events || []; $('#demo-events').innerHTML = events.map(e => `<span class="event-item ${e.played ? 'played' : ''}">${e.played ? '✓ ' : ''}${escapeHtml(e.description)}</span>`).join('') || '<span class="muted-note">No scripted events.</span>'; }
function render() { if (!state.data) return; $('#step-indicator').textContent = `Step ${state.data.step}`; $('#models-toggle').textContent = `Models ${state.data.use_models ? 'on' : 'off'}`; renderBeds(); renderPatients(); renderTimeline(); renderCycle(state.currentCycle); $('#disclaimer').textContent = state.meta?.disclaimer || ''; renderDemo(); }
function setMode(mode) { state.mode = mode; const admit = mode === 'admit'; $('#workspace').classList.toggle('hidden', admit); $('#scenario-strip').classList.toggle('hidden', admit); $('#intake-panel').classList.toggle('hidden', !admit); $('#scenario-tab').classList.toggle('active', !admit); $('#admit-tab').classList.toggle('active', admit); $('#scenario-tab').setAttribute('aria-selected', String(!admit)); $('#admit-tab').setAttribute('aria-selected', String(admit)); }
function showResourceMenu(event, rid) { event.stopPropagation(); const resource = state.data.resources.find(r => r.id === rid); if (!resource) return; const menu = $('#resource-menu'); const out = resource.status === 'out_of_service'; menu.innerHTML = `<button role="menuitem" data-state="${out ? 'in' : 'out'}">${out ? 'Back in service' : 'Out of service'}</button>`; menu.style.left = `${Math.min(event.clientX, innerWidth - 180)}px`; menu.style.top = `${Math.min(event.clientY, innerHeight - 70)}px`; menu.classList.remove('hidden'); menu.querySelector('button').onclick = async () => { menu.classList.add('hidden'); await request(`/api/resources/${encodeURIComponent(rid)}/${out ? 'in' : 'out'}`, {}, '#action-error'); }; }
function openDrawer() { $('#drawer-backdrop').classList.remove('hidden'); $('#patient-drawer').classList.add('open'); $('#patient-drawer').setAttribute('aria-hidden','false'); $('#drawer-close').focus(); }
function closeDrawer() { $('#drawer-backdrop').classList.add('hidden'); $('#patient-drawer').classList.remove('open'); $('#patient-drawer').setAttribute('aria-hidden','true'); state.selectedPid = null; }
async function openPatient(pid) { try { const p = await api(`/api/patients/${encodeURIComponent(pid)}`); state.selectedPid = pid; renderDrawer(p); openDrawer(); } catch (e) { toast(e.message); } }
function probabilityBars(probabilities) { if (!probabilities) return ''; return Object.entries(probabilities).flatMap(([model, values]) => values && typeof values === 'object' ? Object.entries(values).map(([level,prob]) => `<div class="bar-row"><span>${escapeHtml(model.toUpperCase())} ${escapeHtml(level)}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${Math.max(0,Math.min(100,Number(prob)*100))}%"></span></span><span>${Math.round(Number(prob)*100)}%</span></div>`) : []).join(''); }
function renderDrawer(p) {
  const needs = p.needs || {}; const allocation = p.allocation;
  $('#drawer-content').innerHTML = `<h2 class="drawer-title"><span class="mono">${escapeHtml(p.pid)}</span> · Patient details</h2><div class="drawer-subtitle">${escapeHtml(p.age)} years${p.sex ? ` · ${escapeHtml(p.sex)}` : ''} · <span class="risk-chip">${riskDot(p.risk)}${escapeHtml(p.risk || 'Unrated')}</span></div><section class="drawer-section"><h3>Model estimates</h3>${probabilityBars(p.probabilities) || '<p class="muted-note">No model estimates available.</p>'}</section><section class="drawer-section"><h3>Clinical reasoning</h3><pre class="trace">${escapeHtml(p.explanation || 'No deduction tree available.')}</pre><h3>Resolution proof</h3><pre class="trace">${escapeHtml(p.proof || 'No proof available.')}</pre></section><section class="drawer-section"><h3>Needs and allocation</h3><p>${escapeHtml(listText(needs))}</p><p>Allocation: ${escapeHtml(listText(allocation || 'Not allocated'))}</p><p>Held resources: ${escapeHtml(listText(p.holds || []))}</p></section><section class="drawer-section"><h3>Record new vitals</h3><form id="vitals-form" class="vitals-form">${(state.meta?.vitals || []).map(v => `<label>${escapeHtml(v.label)} (${escapeHtml(v.unit || '')})<input type="number" name="${escapeHtml(v.key)}" min="${escapeHtml(v.min)}" max="${escapeHtml(v.max)}" step="any"></label>`).join('')}<button class="primary-button" type="submit">Record vitals</button></form><p class="inline-error" id="vitals-error" role="alert"></p></section><section class="drawer-section"><div class="drawer-actions"><button class="quiet-button" id="discharge-button">Discharge patient</button></div></section>`;
  $('#vitals-form').addEventListener('submit', async e => { e.preventDefault(); const values = Object.fromEntries(new FormData(e.currentTarget).entries()); const body = Object.fromEntries(Object.entries(values).filter(([,v])=>v!=='').map(([k,v])=>[k,Number(v)])); if (!Object.keys(body).length) { $('#vitals-error').textContent='Enter at least one vital sign.'; return; } const result=await request(`/api/patients/${p.pid}/vitals`,body,'#vitals-error'); if(result) { closeDrawer(); toast(`Vitals recorded for ${p.pid}`); } });
  $('#discharge-button').addEventListener('click', async () => { if (!confirm(`Discharge ${p.pid}?`)) return; const result=await request(`/api/patients/${p.pid}/discharge`,{},'#action-error'); if(result) closeDrawer(); });
}
function initializeIntake() {
  const m = state.meta; $('#symptom-options').innerHTML = (m.symptoms || []).map(value => `<label class="check-label"><input type="checkbox" name="symptoms" value="${escapeHtml(value)}"> ${escapeHtml(titleCase(value))}</label>`).join('');
  $('#condition-options').innerHTML = (m.conditions || []).map(value => `<label class="check-label"><input type="checkbox" name="conditions" value="${escapeHtml(value)}"> ${escapeHtml(titleCase(value))}</label>`).join('');
  $('#vital-fields').innerHTML = (m.vitals || []).map(v => `<label>${escapeHtml(v.label)} <span class="unit">${escapeHtml(v.unit || '')}</span><input type="number" name="vital_${escapeHtml(v.key)}" min="${escapeHtml(v.min)}" max="${escapeHtml(v.max)}" step="any" aria-label="${escapeHtml(v.label)} between ${escapeHtml(v.min)} and ${escapeHtml(v.max)} ${escapeHtml(v.unit || '')}"></label>`).join('');
  const historyCols=m.history_columns||[]; $('#history-head').innerHTML=`<tr>${historyCols.map(c=>`<th>${escapeHtml(c.key.toUpperCase())}</th>`).join('')}<th>Remove</th></tr>`; $('#history-limit').textContent=m.history_max_rows || 9;
}
function addHistoryRow(values={}) { const max=state.meta?.history_max_rows || 9; if ($('#history-body').children.length >= max) { toast(`Maximum ${max} earlier hours`); return; } const cols=state.meta.history_columns||[]; const row=document.createElement('tr'); row.innerHTML=cols.map(c=>`<td><input type="number" data-history="${escapeHtml(c.key)}" min="${escapeHtml(c.min)}" max="${escapeHtml(c.max)}" step="any" aria-label="${escapeHtml(c.key)} value" value="${values[c.key] ?? ''}"></td>`).join('')+'<td><button type="button" aria-label="Remove hour">×</button></td>'; row.querySelector('button').addEventListener('click',()=>row.remove()); $('#history-body').append(row); }
function loadExample() {
  const form=$('#admit-form'); form.elements.age.value='78'; form.elements.sex.value='f';
  ['shortness_of_breath','confusion'].forEach(v=>{const el=form.querySelector(`[name="symptoms"][value="${v}"]`); if(el)el.checked=true;});
  ['COPD','Diabetes','copd','diabetes'].forEach(v=>{const el=form.querySelector(`[name="conditions"][value="${v}"]`); if(el)el.checked=true;});
  const sample={hr:124,sbp:92,dbp:54,spo2:89,temp:38.9,rr:28,lactate:3.4}; Object.entries(sample).forEach(([k,v])=>{const input=form.querySelector(`[name="vital_${k}"]`);if(input)input.value=v;});
  $('#history-body').replaceChildren(); const sampleRows=[{hr:102,map:75,spo2:94,temp:37.3,rr:21,lactate:1.8},{hr:108,map:72,spo2:93,temp:37.6,rr:22,lactate:2.0},{hr:114,map:68,spo2:92,temp:38.0,rr:24,lactate:2.4},{hr:119,map:64,spo2:90,temp:38.5,rr:26,lactate:2.9},{hr:124,map:60,spo2:89,temp:38.9,rr:28,lactate:3.4}]; sampleRows.forEach(row=>addHistoryRow(row));
}
function renderAdmission(result) { const probs=result.probabilities||{}; const estimates=Object.entries(probs).filter(([k,v])=>k!=='ml_features_known'&&v).map(([model,values])=>`<div class="drawer-section"><h3>${escapeHtml(model.toUpperCase())} estimates</h3>${probabilityBars({[model]:values})}</div>`).join(''); const cycle=result.cycle||{}; const notices=cycle.search?.astar?.announcements||[];
  $('#admit-result').classList.remove('hidden'); $('#admit-result').innerHTML=`<h2>Admitted <span class="patient-id">${escapeHtml(result.pid)}</span></h2><p>Rules-derived risk: <strong>${escapeHtml(result.state?.patients?.find(p=>p.pid===result.pid)?.risk || 'See queue')}</strong></p>${probs.ml_features_known!=null?`<p>The ML model knew ${Math.round(probs.ml_features_known*100)} % of its inputs.</p>`:''}${estimates}<div class="drawer-section"><h3>Allocation changes</h3><pre class="trace">${escapeHtml(listText(cycle.allocation_changes||cycle.allocation||'See agent panel'))}</pre>${notices.map(n=>`<div class="warning-block">! ${escapeHtml(typeof n==='string'?n:listText(n))}</div>`).join('')}</div><button class="quiet-button" id="view-admitted">View ${escapeHtml(result.pid)} in queue</button>`;
  $('#view-admitted').addEventListener('click',()=>{setMode('scenario');document.querySelector(`[data-patient="${CSS.escape(result.pid)}"]`)?.scrollIntoView({behavior:'smooth',block:'center'});openPatient(result.pid);});
}
async function submitAdmission(event) { event.preventDefault(); const form=event.currentTarget; const error=$('#admit-error'); error.textContent=''; const fd=new FormData(form); const age=Number(fd.get('age')); if(!Number.isFinite(age)||age<0||age>120){error.textContent='Age must be between 0 and 120.';return;} const vitals={}; for(const v of state.meta.vitals||[]){const raw=fd.get(`vital_${v.key}`);if(raw==='')continue;const num=Number(raw);if(!Number.isFinite(num)||num<v.min||num>v.max){error.textContent=`${v.label} must be between ${v.min} and ${v.max} ${v.unit||''}.`;return;}vitals[v.key]=num;}
  const history=[]; for(const row of $$('#history-body tr')){const entry={}; row.querySelectorAll('[data-history]').forEach(input=>{if(input.value!=='')entry[input.dataset.history]=Number(input.value);});if(Object.keys(entry).length)history.push(entry);}
  const body={age,sex:fd.get('sex')||null,symptoms:fd.getAll('symptoms'),conditions:fd.getAll('conditions'),vitals,low_bp:fd.has('low_bp'),history};
  const result=await request('/api/patients',body,'#admit-error'); if(result){renderAdmission(result);}
}
async function runAction(type) { $('#action-error').textContent=''; if(type==='step') return request('/api/step',{},'#action-error'); if(type==='three')return request('/api/run',{steps:3},'#action-error'); if(type==='transfer')return request('/api/run',{until:'transfer'},'#action-error'); return request('/api/run',{all:true},'#action-error'); }
async function refreshReference() { try { state.meta=await api('/api/meta'); initializeIntake(); } catch(e) { $('#connection-status').innerHTML=`<span class="status-dot"></span> Agent unavailable`; toast(e.message); throw e; } }
async function loadInitial() { await refreshReference(); const [scenarios, data, demo]=await Promise.all([api('/api/scenarios'),api('/api/state'),api('/api/demo')]); state.scenarios=scenarios.scenarios||[];state.data=data;state.demo=demo;state.currentCycle=data.cycles?.length?await api(`/api/cycles/${data.cycles[data.cycles.length-1].number}`):null; $('#scenario-picker').innerHTML=state.scenarios.map(s=>`<option value="${escapeHtml(s.file)}" ${s.file===scenarios.current?'selected':''}>${escapeHtml(s.name)}</option>`).join(''); $('#connection-status').innerHTML='<span class="status-dot online"></span> Agent connected'; render(); }
async function reset(scenario=state.data?.scenario?.file, useModels=state.data?.use_models, empty=false) { const result=await request('/api/reset',{scenario,use_models:useModels,empty},'#action-error'); if(result){state.demo=await api('/api/demo').catch(()=>state.demo);state.currentCycle=result.cycle;render();if(empty)setMode('scenario');$('#admit-result').classList.add('hidden');} }
$('#scenario-tab').addEventListener('click',()=>setMode('scenario')); $('#admit-tab').addEventListener('click',()=>setMode('admit'));
$('#scenario-picker').addEventListener('change',e=>reset(e.target.value,state.data?.use_models)); $('#reset-button').addEventListener('click',()=>reset()); $('#models-toggle').addEventListener('click',()=>reset(state.data?.scenario?.file,!state.data?.use_models));
$('#demo-next').addEventListener('click',async()=>{const result=await request('/api/demo/next',{},'#demo-error');if(result){state.demo=await api('/api/demo').catch(()=>state.demo);renderDemo();}});
$$('[data-run]').forEach(b=>b.addEventListener('click',()=>runAction(b.dataset.run)));
$('#drawer-close').addEventListener('click',closeDrawer); $('#drawer-backdrop').addEventListener('click',closeDrawer); document.addEventListener('keydown',e=>{if(e.key==='Escape'){closeDrawer();$('#resource-menu').classList.add('hidden');}});document.addEventListener('click',e=>{if(!e.target.closest('[data-resource]')&&!e.target.closest('#resource-menu'))$('#resource-menu').classList.add('hidden');});
$('#load-example').addEventListener('click',loadExample); $('#add-history').addEventListener('click',()=>addHistoryRow()); $('#admit-form').addEventListener('submit',submitAdmission); $('#empty-department').addEventListener('click',()=>reset(state.data?.scenario?.file,state.data?.use_models,true));
$('#query-form').addEventListener('submit',async e=>{e.preventDefault();const text=$('#query-input').value.trim();try{const result=await api('/api/query',{method:'POST',body:JSON.stringify({text})});const out=$('#query-output');if(result.kind==='proof')out.textContent=result.text;else out.innerHTML=`<table><thead><tr>${result.variables.map(v=>`<th>${escapeHtml(v)}</th>`).join('')}</tr></thead><tbody>${result.rows.map(row=>`<tr>${result.variables.map(v=>`<td>${escapeHtml(row[v]??'')}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}catch(err){$('#query-output').textContent=err.message;}});
$$('[data-query]').forEach(b=>b.addEventListener('click',()=>{$('#query-input').value=b.dataset.query;$('#query-form').requestSubmit();}));
// Colour theme: light, dark, or auto (follows the operating system). The class
// sits on <html> so the whole page, form controls and scrollbars switch.
const darkQuery = matchMedia('(prefers-color-scheme: dark)');
function applyTheme(choice) {
  if (!['light', 'dark', 'auto'].includes(choice)) choice = 'light';
  try { localStorage.setItem('ward-theme', choice); } catch {}
  const dark = choice === 'dark' || (choice === 'auto' && darkQuery.matches);
  document.documentElement.classList.toggle('dark', dark);
  document.documentElement.dataset.theme = choice;
  $$('.theme-switch button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.theme === choice)));
}
$$('.theme-switch button').forEach(b => b.addEventListener('click', () => applyTheme(b.dataset.theme)));
darkQuery.addEventListener('change', () => { if (document.documentElement.dataset.theme === 'auto') applyTheme('auto'); });
let savedTheme = 'light'; try { savedTheme = localStorage.getItem('ward-theme') || 'light'; } catch {}
applyTheme(savedTheme);
loadInitial().catch(error=>{ $('#connection-status').innerHTML=`<span class="status-dot"></span> Agent unavailable`; $('#agent-content').innerHTML=`<div class="empty-state">Could not connect to the agent: ${escapeHtml(error.message)}. Start the Python server and reload.</div>`; });
