"""Read-only RSI status presentation for terminals and a loopback browser."""
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread


LABELS = {"repeated": "Repeated patients", "novel_same_department": "New patients / same dept",
          "novel_departments": "Transfer / five depts"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def status_data(manifest):
    import psutil
    owner = manifest.get("owner", {})
    try:
        alive = psutil.Process(owner["pid"]).create_time() == owner["created"]
    except (KeyError, psutil.Error):
        alive = False
    now = datetime.now(timezone.utc)
    jobs = []
    for condition, job in manifest.get("jobs", {}).items():
        progress = read_json(Path(job["output"]) / "progress.json", {})
        state = job["status"]
        if state == "running" and not alive:
            state = "launcher offline"
        try:
            age = max(0, (now - datetime.fromisoformat(progress["updated_at"])).total_seconds())
        except (KeyError, ValueError, TypeError):
            age = None
        planned = manifest.get("preflight", {}).get("conditions", {}).get(condition, {})
        training_total = progress.get("training_total", planned.get("train_exposure_count", 250))
        test_total = progress.get("test_total", planned.get("primary_test_count", 0) * len(planned.get("checkpoints", ()))
                                  + planned.get("advanced_test_count", 0))
        worker_alive = None
        if job.get("pid"):
            try:
                worker_alive = psutil.Process(job["pid"]).create_time() == job["created"]
            except (KeyError, psutil.Error):
                worker_alive = False
        jobs.append({"condition": condition, "label": LABELS.get(condition, condition),
                     "status": state, "progress": progress,
                     "attempt": job.get("attempt"), "worker_alive": worker_alive,
                     "worker_log": str(Path(job["config"]).parent / "worker.log") if job.get("config") else None,
                     "heartbeat_age_s": age, "stale": state == "running" and age is not None and age > 30,
                     "training_total": training_total, "test_total": test_total,
                     "completed": progress.get("training_completed", 0) + progress.get("test_completed", 0),
                     "stop_target": manifest.get("stop_targets", {}).get(condition)})
    summaries = [job["progress"].get("usage", {}) for job in jobs]
    totals = {key: sum(row.get(key, 0) or 0 for row in summaries) for key in (
        "llm_calls", "embedding_calls", "known_total_tokens", "failed_calls", "untracked_calls")}
    totals["token_coverage"] = "complete" if summaries and all(row.get("token_coverage") == "complete" for row in summaries) else "partial/unknown"
    return {"status": manifest.get("status", "starting") if alive else
            "launcher offline" if manifest.get("status") == "running" else manifest.get("status", "starting"),
            "owner_alive": alive, "jobs": jobs, "usage": totals,
            "completed": sum(job["completed"] for job in jobs),
            "planned_encounters": 22 if manifest.get("pilot") else 20 if manifest.get("smoke") else 1100}


def print_status(manifest):
    def percent(value):
        return "--" if value is None else f"{value * 100:.1f}%"
    def count(value):
        return f"{value:,}"
    data = status_data(manifest)
    lines = [f"Mem0 RSI | {data['status'].upper()} | {datetime.now():%H:%M:%S}",
             f"Overall {data['completed']}/{data['planned_encounters']} ({data['completed']/data['planned_encounters']:.1%})", "-" * 78]
    for job in data["jobs"]:
        p = job["progress"]
        usage = p.get("usage", {})
        test = p.get("test_metrics", {})
        phase = p.get("phase", "starting")
        train_done, test_done = p.get("training_completed", 0), p.get("test_completed", 0)
        heartbeat = "--" if job["heartbeat_age_s"] is None else f"{job['heartbeat_age_s']:.0f}s"
        lines.append(f"{job['label']} | {job['status']} | {phase}")
        lines.append(f"  Train {train_done}/{p.get('training_total', '?')}   Test {test_done}/{p.get('test_total', '?')}"
                     f"   Current {p.get('patient_id', '--')}   {p.get('elapsed_s', 0) / 60:.1f} min")
        lines.append(f"  Attempt {job['attempt']}   Current case {p.get('case_elapsed_s', 0) or 0:.0f}s"
                     f"   Checkpoint {p.get('checkpoint', '--')}   Heartbeat age {heartbeat}")
        if job["stale"] or job["worker_alive"] is False and job["status"] == "running":
            lines.append("  ATTENTION: worker exited or progress heartbeat is older than 30 seconds.")
        if p.get("failure"):
            lines.append(f"  {p['failure']['type']}: {p['failure']['message']}")
        metrics = test if phase in {"primary_test", "advanced_test"} else p.get("training_metrics", {})
        name = "Test" if phase in {"primary_test", "advanced_test"} else "Train"
        if metrics.get("evaluated_cases", 0):
            lines.append(f"  {name}: Dx {percent(metrics.get('diagnosis_accuracy'))}"
                         f"   Exam P/R {percent(metrics.get('examination_precision'))}/{percent(metrics.get('examination_recall'))}"
                         f"   Treatment {percent(metrics.get('treatment_overall_score'))}")
        else:
            lines.append(f"  {name}: awaiting first completed case")
        lines.append(f"  Calls {count(usage.get('llm_calls', 0))} LLM / {count(usage.get('embedding_calls', 0))} embed"
                     f"   Known tokens {count(usage.get('known_total_tokens', 0))}"
                     f"   Cache {percent(usage.get('prompt_cache_hit_rate'))}"
                     f"   Errors {usage.get('failed_calls', 0)}")
        recovery = p.get("json_recovery", {})
        lines.append(f"  JSON repairs {recovery.get('repairs', 0)} / fallbacks {recovery.get('fallbacks', 0)}"
                     f"   Memories {p.get('memory', {}).get('memories', '--')}"
                     f"   Learned {p.get('learning', {}).get('learned_exposures', '--')}")
        lines.append("")
    lines.append("Scores appear after a case finishes. -- = not reported yet.")
    if manifest.get("pilot"):
        lines.append("Pilot: 20 training cases, restart after 10, then 2 frozen tests.")
    elif any(job["condition"] == "novel_departments" for job in data["jobs"]):
        lines.append("Transfer starts with baseline tests before learning. More details: browser monitor.")
    print("\n".join(lines), flush=True)


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mem0 RSI - Live progress</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#dce7f6;background:#0d1422}*{box-sizing:border-box}
body{margin:0;padding:30px;max-width:1450px;margin:auto}h1{margin:0 0 8px;font-size:28px}h2{font-size:19px;margin:0}
.muted{color:#9eafc8;font-size:14px;line-height:1.6}.top{display:flex;justify-content:space-between;gap:18px;align-items:center}
.badge{background:#203249;border-radius:20px;padding:6px 12px;font-size:13px;white-space:nowrap}.running{color:#6fe5b2}.failed,.offline{color:#ffb18d}
#cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;margin:24px 0}.card{background:#152136;border:1px solid #2c3e59;border-radius:14px;padding:20px}
.phase{font-size:16px;margin:20px 0 10px;color:#79bbff}.big{font-size:30px;font-weight:650;margin:8px 0}
progress{width:100%;height:12px;accent-color:#62b7ff}.row{display:flex;justify-content:space-between;gap:12px;padding:7px 0;font-size:14px}
.section{border-top:1px solid #30415a;margin-top:15px;padding-top:12px}.section h3{font-size:13px;text-transform:uppercase;color:#a4b5cc;letter-spacing:.07em;margin:0 0 5px}
.hint{background:#18283e;border-radius:8px;padding:12px;font-size:14px;line-height:1.6;color:#b8cbe3}.empty{color:#afc1d9;font-size:14px;padding:12px 0}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:22px 0}.summary>div{background:#152136;border:1px solid #2c3e59;border-radius:12px;padding:18px}.summary strong{display:block;font-size:25px;margin-top:7px}
.warning{color:#ffd192}.attention{border-color:#dc9e56}.meter{margin:16px 0}table{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0}th,td{padding:8px 5px;text-align:right;border-bottom:1px solid #30415a}th:first-child,td:first-child{text-align:left}.scroll{overflow-x:auto}.small{font-size:12px}.completed{color:#6fe5b2}
details{margin-top:12px}summary{cursor:pointer;font-size:14px;color:#80bfff}#error{color:#ffbc9d;margin:15px 0}footer{font-size:13px;color:#99aac1;line-height:1.7;overflow-wrap:anywhere}
@media(max-width:1150px){#cards{grid-template-columns:1fr}.summary{grid-template-columns:repeat(2,1fr)}body{padding:18px}.big{font-size:26px}}
</style><div class="top"><div><h1>Mem0 x OpenHospital RSI</h1><div class="muted" id="configuration">Shared hospital · Seed 0</div></div><span class="badge" id="state">Connecting...</span></div>
<div id="error"></div><div id="total" class="big">Loading progress...</div><div class="muted" id="updated"></div>
<progress id="overall-progress" class="meter" value="0" max="1100"></progress>
<div class="summary" id="summary"></div>
<div id="cards"></div><div class="hint"><span id="schedule"></span> Scores appear after each completed case. <b>Not reported</b> does not mean zero. This page refreshes every 3 seconds and only reads progress files. Closing it will not stop the experiment.</div>
<footer><p id="directory"></p><p>Use the stop command to finish active cases and save. The browser monitor does not control, restart, or modify workers.</p></footer>
<script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n=x=>Number(x||0).toLocaleString('en-US');const pct=x=>x==null?'Not reported':(x*100).toFixed(1)+'%';
const row=(a,b)=>`<div class="row"><span>${esc(a)}</span><strong>${esc(b)}</strong></div>`;
const duration=x=>x==null?'Not reported':x>=3600?(x/3600).toFixed(1)+' h':x>=60?(x/60).toFixed(1)+' min':x.toFixed(0)+' s';
function usageTable(rows){if(!rows||!Object.keys(rows).length)return '<div class="empty">No source breakdown recorded yet.</div>';
return '<div class="scroll"><table><thead><tr><th>Source</th><th>Calls</th><th>Known in</th><th>Known out</th><th>Errors</th></tr></thead><tbody>'+Object.entries(rows).map(([label,r])=>`<tr><td>${esc(label)}${r.token_coverage!=='complete'?' *':''}</td><td>${n(r.tracked_calls)}</td><td>${n(r.prompt_tokens)}</td><td>${n(r.completion_tokens)}</td><td>${n(r.failed_calls)}</td></tr>`).join('')+'</tbody></table></div><div class="muted small">* Some request usage is unknown. Hospital simulation usage updates after each case.</div>'}
function checkpoints(rows){if(!rows?.length)return '';
return '<div class="section"><h3>Completed frozen checkpoints</h3><div class="scroll"><table><thead><tr><th>Train / split</th><th>Scored</th><th>Diagnosis</th><th>Exam P</th><th>Treatment</th></tr></thead><tbody>'+[...rows].sort((a,b)=>a.checkpoint-b.checkpoint).map(r=>`<tr><td>${n(r.checkpoint)} / ${r.split==='advanced_test'?'Advanced':'Primary'}</td><td>${n(r.evaluated_cases)}</td><td>${pct(r.diagnosis_accuracy)}</td><td>${pct(r.examination_precision)}</td><td>${pct(r.treatment_overall_score)}</td></tr>`).join('')+'</tbody></table></div></div>'}
const phaseName={train:'Training',primary_test:'Frozen primary test',advanced_test:'Frozen advanced test',completed:'Completed',starting:'Starting services'};
const departmentName={'血液科':'Hematology','呼吸内科':'Respiratory medicine','消化内科':'Gastroenterology','内分泌科':'Endocrinology','神经内科':'Neurology'};
function metrics(title,m){if(!m||!m.evaluated_cases)return `<div class="section"><h3>${title}</h3><div class="empty">Waiting for the first completed case.</div></div>`;
return `<div class="section"><h3>${title} · ${n(m.evaluated_cases)} scored</h3>`+
row('Diagnosis accuracy',pct(m.diagnosis_accuracy))+row('Exam precision / recall',pct(m.examination_precision)+' / '+pct(m.examination_recall))+
row('Treatment score',pct(m.treatment_overall_score))+row('Overall clinical score',pct(m.overall_clinical_score))+
row('Scoring coverage',`${n(m.evaluated_cases)} / ${n(m.expected_cases)} (${pct(m.evaluation_coverage)})`)+
`<details data-key="${esc(title)}"><summary>Treatment details</summary>${row('Safety',pct(m.treatment_safety))}${row('Effectiveness',pct(m.treatment_effectiveness_alignment))}${row('Personalization',pct(m.treatment_personalization))}${row('No examination',pct(m.no_examination_rate))}</details></div>`}
function card(j){let p=j.progress,u=p.usage||{},m=p.memory||{},l=p.learning||{},train=p.training_completed||0,test=p.test_completed||0,tt=j.training_total,ht=j.test_total,done=train+test,total=tt+ht;
let phase=phaseName[p.phase]||p.phase||'Starting';if(p.phase==='primary_test'&&p.checkpoint===0)phase='Baseline test (before training)';
return `<article data-condition="${esc(j.condition)}" class="card ${j.stale?'attention':''}"><div class="top"><h2>${esc(j.label)}</h2><span class="${esc(j.status)} badge">${esc(j.status)}</span></div><div class="phase">${esc(phase)}</div><div class="big">${n(done)} <span class="muted">/ ${n(total)} cases · ${(total?done/total*100:0).toFixed(1)}%</span></div><progress value="${done}" max="${total}"></progress>`+
row('Training',`${train} / ${tt}`)+row('Frozen held-out tests',ht?`${test} / ${ht}`:'Not used in this condition')+row('Current patient',p.patient_id||'Waiting for worker')+row('Current case elapsed',duration(p.case_elapsed_s))+row('Current attempt elapsed',duration(p.elapsed_s))+
(p.phase==='primary_test'||p.phase==='advanced_test'?row('Parallel tests',`${p.active_test_count??0} active / ${p.test_concurrency??1} slots`)+row('Active test patients',(p.active_test_patients||[]).join(', ')||'Between cases')+row('Current test panel',`${p.test_batch_completed??0} / ${p.test_batch_total??50}`):'')+
row('Department',departmentName[p.department]||p.department||'Not reported')+row('Training block / exposure',`${p.batch??'--'} / ${p.position??'--'}`)+row('Attempt / last checkpoint',`${j.attempt??'--'} / ${p.checkpoint??'--'}`)+row('Last 5 training mean',duration(p.timing?.mean_last_5_training_s))+
(j.stale?'<div class="hint warning">No worker heartbeat for over 30 seconds. Check worker.log; this does not prove the case has failed.</div>':'')+
(j.worker_alive===false&&j.status==='running'?'<div class="hint warning">Worker process has exited. Waiting for launcher status.</div>':'')+
(j.stop_target?row('Automatic stop',`Train ${j.stop_target.training} · Tests ${j.stop_target.tests}`):'')+
(p.pause_reason?`<div class="hint">${esc(p.pause_reason)}</div>`:'')+
(p.failure?`<div class="hint warning">${esc(p.failure.type)}: ${esc(p.failure.message)}</div>`:'')+
metrics('Training scores',p.training_metrics)+(ht?metrics('Latest test checkpoint',p.test_metrics):'')+
checkpoints(p.test_checkpoints)+
`<div class="section"><h3>Usage & health</h3>`+row('LLM / embedding calls',`${n(u.llm_calls)} / ${n(u.embedding_calls)}`)+row('Known tokens',n(u.known_total_tokens))+
row('Prompt cache hit rate',pct(u.prompt_cache_hit_rate))+row('Failed API calls',n(u.failed_calls))+row('Learned / pending feedback',`${n(l.learned_exposures)} / ${n(l.pending_feedback)}`)+
row('JSON repairs / fallbacks',`${n(p.json_recovery?.repairs)} / ${n(p.json_recovery?.fallbacks)}`)+
`<details open data-key="usage"><summary>Requests by source</summary>${usageTable(p.usage_breakdown)}</details>`+
`<details data-key="accounting"><summary>Memory & accounting</summary>`+row('Mem0 memories',n(m.memories))+
row('Cached / cache-created tokens',`${n(u.known_cached_prompt_tokens)} / ${n(u.known_cache_creation_prompt_tokens)}`)+row('Cache reporting coverage',u.cache_token_coverage||'unknown')+
row('Token reporting coverage',u.token_coverage||'unknown')+row('Recorded cost',u.cost==null?'Price not configured':u.cost+' '+(u.currency||''))+row('Saved after training case',p.latest_snapshot?.position??'Not yet')+
row('Heartbeat age',duration(j.heartbeat_age_s))+row('Last worker update',p.updated_at?new Date(p.updated_at).toLocaleTimeString():'Waiting')+`<p class="muted small">Worker log: ${esc(j.worker_log||'Waiting for launch')}</p></details></div></article>`}
async function refresh(){try{let response=await fetch('/status',{cache:'no-store'});if(!response.ok)throw Error('Status unavailable');let d=await response.json();
document.getElementById('state').textContent=d.status.toUpperCase();document.getElementById('state').className='badge '+(d.status==='completed'?'completed':d.owner_alive?'running':d.status==='starting'?'':'offline');
let done=d.completed;document.getElementById('total').textContent=n(done)+' / '+n(d.planned_encounters)+' encounters completed · '+(100*done/d.planned_encounters).toFixed(1)+'%';
document.getElementById('overall-progress').value=done;document.getElementById('overall-progress').max=d.planned_encounters;
let u=d.usage;document.getElementById('summary').innerHTML=[['LLM requests',n(u.llm_calls)],['Embedding requests',n(u.embedding_calls)],['Known tokens',n(u.known_total_tokens)],['Failed / unknown usage',n(u.failed_calls)+' / '+n(u.untracked_calls)]].map(([label,value])=>`<div><span class="muted">${esc(label)}</span><strong>${esc(value)}</strong></div>`).join('');
document.getElementById('updated').textContent='Last refresh: '+new Date().toLocaleTimeString()+' · Overall status: '+d.status;
document.getElementById('configuration').textContent=(d.planned_encounters===22?'Pilot · One independent memory store':'Three independent conditions')+' · Shared hospital · Seed 0';
document.getElementById('schedule').textContent=d.planned_encounters===22?'Pilot: 20 training cases, restart after 10, then 2 frozen tests.':d.jobs.some(j=>j.condition==='novel_departments')?'The transfer condition begins with 50 baseline tests before training.':'';
const detailKey=el=>el.closest('article').dataset.condition+'/'+el.dataset.key;
const opened=new Map([...document.querySelectorAll('details[data-key]')].map(el=>[detailKey(el),el.open]));
document.getElementById('cards').innerHTML=d.jobs.map(card).join('')||'<div class="hint">Starting services and provider checks. Waiting for suite.json; no encounter has started yet.</div>';
document.querySelectorAll('details[data-key]').forEach(el=>{if(opened.has(detailKey(el)))el.open=opened.get(detailKey(el))});
document.getElementById('directory').textContent='Results: '+d.run_dir;document.getElementById('error').textContent='';
}catch(e){document.getElementById('error').textContent='Monitor connection lost. The experiment may still be running. '+e.message}}
refresh();setInterval(refresh,3000);
</script></html>'''


def make_server(run_dir, port=8767):
    run = Path(run_dir).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                payload, kind = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            elif self.path == "/status":
                manifest = read_json(run / "suite.json", {})
                payload = json.dumps({**status_data(manifest), "run_dir": str(run)}).encode("utf-8")
                kind = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


@contextmanager
def background_monitor(run_dir, port=8767):
    """Own a read-only monitor for a launcher's lifetime, including preflight."""
    server = make_server(run_dir, port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def serve(run_dir, port=8767):
    with make_server(run_dir, port) as server:
        print(f"Read-only progress: http://127.0.0.1:{port}", flush=True)
        server.serve_forever()
