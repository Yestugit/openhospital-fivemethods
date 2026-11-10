"""Presentation must not turn missing or stale data into measured success."""
from io import StringIO
import json

from mematlas.integrations.openhospital.dashboard import collect, render, Console, duration


def fixture(tmp_path, *, completed=3, status="running"):
    output = tmp_path / "memos/attempt-001/evaluation"
    output.mkdir(parents=True)
    (output / "progress.json").write_text(json.dumps({"training_completed":completed,
        "patient_id":"Patient_next", "phase":"train", "latest_snapshot":{"position":completed},
        "training_metrics":{"diagnosis_accuracy":.5,"treatment_overall_score":.625,
            "evaluation_coverage":1,"evaluated_cases":completed},
        "round_metrics":{"diagnosis_accuracy":.5,"treatment_overall_score":.625},
        "usage":{"llm_calls":20,"embedding_calls":5,"failed_calls":0},
        "json_recovery":{"repairs":1,"fallbacks":0}}),encoding="utf-8")
    snapshot=output/'runs/001-memos/latest_snapshot.json'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_text(json.dumps({'position':completed}),encoding='utf-8')
    manifest={"status":status,"settings":{"smoke":False},"jobs":{
        "memos":{"status":"running","output":str(output),"position":0,"attempt":1},
        **{m:{"status":"pending","position":0} for m in ("mem0","memmachine","emem","memora")}}}
    (tmp_path / "suite.json").write_text(json.dumps(manifest),encoding="utf-8")
    folder=tmp_path / "memos/activity";folder.mkdir()
    (folder / "measurement-0.json").write_text(json.dumps({"round":0,"rsi_tokens":0,"payload_tokens":0,
        "memory_text_tokens":0,"storage_bytes":1048576}),encoding="utf-8")
    return output,folder


def test_readable_screen_has_no_raw_json_and_labels_baseline(tmp_path):
    fixture(tmp_path)
    state=collect(tmp_path)
    assert state["completed"]==3 and state["total"]==500
    screen=render(state)
    assert "3/500" in screen and "50.0%" in screen and "0.625" in screen
    assert "EVERY CASE" in screen
    assert "MEMORY method time" in screen and "Saved checkpoint: case 3" in screen
    assert '"metric_counts"' not in screen and "{" not in screen
    assert max(map(len,screen.splitlines())) < 110
    assert len(screen.splitlines()) <= 30


def test_redirected_console_is_plain_text(tmp_path):
    fixture(tmp_path)
    stream=StringIO();Console(stream).show(render(collect(tmp_path)))
    assert "\x1b" not in stream.getvalue()
    assert "MemMachine" in stream.getvalue()
    assert duration(3661)=="01:01:01"


def test_complete_round_keeps_measured_values_and_score_fields(tmp_path):
    output,folder=fixture(tmp_path,completed=20)
    (folder / "measurement-1.json").write_text(json.dumps({"round":1,"rsi_tokens":12345,
        "memory_text_tokens":2345,"storage_bytes":2097152,"round_total_s":600,"memory_method_wall_s":230}),encoding="utf-8")
    curve=output / "runs/001-memos/curve.json";curve.parent.mkdir(parents=True,exist_ok=True)
    curve.write_text(json.dumps([{"batch_index":1,"diagnosis_accuracy":.4,"treatment_overall_score":.7}]),encoding="utf-8")
    state=collect(tmp_path)
    assert state["rounds"][0]["memory_method_wall_s"]==230
    assert state["rounds"][0]["diagnosis_accuracy"]==.4
    assert "end of round 1" in render(state)
    assert state["active"]["measurement"]["rsi_tokens"]==12345


def test_completed_result_view_reads_compact_clinical_rounds(tmp_path):
    output,folder=fixture(tmp_path,completed=100)
    manifest=json.loads((tmp_path/'suite.json').read_text())
    manifest['jobs']['memos']['status']='completed'
    (tmp_path/'suite.json').write_text(json.dumps(manifest),encoding='utf8')
    (folder/'measurement-1.json').write_text(json.dumps({'round':1,'memory_method_wall_s':40}),encoding='utf8')
    (tmp_path/'memos/clinical-rounds.json').write_text(json.dumps([{'batch_index':1,'diagnosis_accuracy':.75}]),encoding='utf8')
    (output/'runs/001-memos/curve.json').write_text('large clinical artifacts need not be parsed for display',encoding='utf8')
    state=collect(tmp_path)
    assert state['rounds'][0]['diagnosis_accuracy']==.75


def test_request_to_stop_is_visible_without_claiming_stopped(tmp_path):
    fixture(tmp_path)
    (tmp_path / "STOP").touch()
    assert "STOPPING AT CASE BOUNDARY" in render(collect(tmp_path))


def test_missing_metrics_are_not_reported_as_zero(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    screen=render(collect(tmp_path))
    assert "Diagnosis" in screen and "not measured yet" in screen
    assert "Storage -- MiB" in screen and "Failed requests --" in screen


def test_retired_emem_progress_is_replaced_by_empty_serial_queue(tmp_path):
    fixture(tmp_path)
    manifest=json.loads((tmp_path/'suite.json').read_text())
    manifest['execution_mode']='parallel_final_two'
    manifest['jobs']['emem']={'status':'paused','position':51,'output':str(tmp_path/'retired')}
    manifest['jobs']['memora']={'status':'running','position':54}
    (tmp_path/'suite.json').write_text(json.dumps(manifest),encoding='utf8')
    (tmp_path/'serial-replacement.json').write_text(json.dumps({'method':'emem','status':'waiting',
        'waiting_for':'memora','job':{'attempt':1,'position':0,'output':str(tmp_path/'fresh')}}),encoding='utf8')
    state=collect(tmp_path)
    emem=next(row for row in state['methods'] if row['id']=='emem')
    assert emem['completed']==0 and emem['status']=='queued'
    assert state['execution_mode']=='serial'
    assert 'Memora first' in render(state)
    replacement=json.loads((tmp_path/'serial-replacement.json').read_text())
    replacement['status']='deferred'
    (tmp_path/'serial-replacement.json').write_text(json.dumps(replacement),encoding='utf8')
    state=collect(tmp_path)
    assert next(row for row in state['methods'] if row['id']=='emem')['completed']==0
    assert next(row for row in state['methods'] if row['id']=='emem')['status']=='deferred'
    assert 'no automatic EMem start' in render(state)
    manifest['status']='stopped'
    for method in ['memos','mem0','memmachine','memora']:
        manifest['jobs'][method]['status']='completed'
    (tmp_path/'suite.json').write_text(json.dumps(manifest),encoding='utf8')
    assert collect(tmp_path)['active']['id']=='memora'


def test_parallel_dashboard_lists_both_and_retains_independent_progress(tmp_path):
    fixture(tmp_path)
    manifest=json.loads((tmp_path/'suite.json').read_text())
    manifest['execution_mode']='parallel_final_two'
    for method in ('memos','mem0','memmachine'):
        manifest['jobs'][method].update(status='completed',position=100)
    for method,completed in [('emem',3),('memora',5)]:
        out=tmp_path/method/'attempt-001/evaluation';out.mkdir(parents=True)
        (out/'progress.json').write_text(json.dumps({'training_completed':completed,'phase':'train'}))
        manifest['jobs'][method].update(status='running',output=str(out))
    state=collect(tmp_path,manifest)
    assert [m['id'] for m in state['active_methods']]==['emem','memora']
    assert state['completed']==308 and state['execution_mode']=='parallel_final_two'
    assert 'PARALLEL: EMem 3/100 running | Memora 5/100 running' in render(state)


def test_resume_counts_do_not_duplicate_committed_position(tmp_path):
    fixture(tmp_path,completed=21)
    data=json.loads((tmp_path / "suite.json").read_text())
    data["jobs"]["memos"]["position"]=20
    state=collect(tmp_path,data)
    assert state["completed"]==21
    assert state["active"]["checkpoint"]==21


def test_case_samples_update_before_round_end_and_preserve_flat_storage(tmp_path):
    from mematlas.core.provenance import json_sha256
    _,folder=fixture(tmp_path,completed=3)
    target=folder/'cases';target.mkdir()
    for position,tokens in [(0,0),(1,100),(2,210),(3,350)]:
        row={"schema":1,"case_position":position,"round":1 if position else 0,
            "rsi_tokens":tokens,"payload_tokens":tokens*2,"memory_text_tokens":50,
            "storage_bytes":1048576,"phase":"case_accepted","background_pending":False}
        row["payload_sha256"]=json_sha256(row)
        (target/f'case-{position:04d}.json').write_text(json.dumps(row),encoding='utf-8')
    state=collect(tmp_path)
    assert len(state['growth'])==4
    assert [p['rsi_tokens'] for p in state['growth']]==[0,100,210,350]
    assert state['active']['measurement']['case_position']==3
    assert state['active']['measurement']['rsi_tokens']==350
    assert 'after case 3/100' in render(state)
    assert not state['rounds']  # Time/score curves still wait for a complete round.


def test_case_growth_report_does_not_require_finished_round(tmp_path,monkeypatch):
    test_case_samples_update_before_round_end_and_preserve_flat_storage(tmp_path)
    from mematlas.integrations.openhospital.efficiency import report
    monkeypatch.setattr('mematlas.integrations.openhospital.efficiency_metrics.request_inventory',lambda path:({},[]))
    assert report(tmp_path)==[]
    cases=json.loads((tmp_path/'efficiency-cases.json').read_text('utf-8'))
    assert len(cases)==4 and cases[-1]['case_position']==3
    assert len({p['storage_bytes'] for p in cases})==1
    assert (tmp_path/'storage_bytes.png').is_file()
    assert (tmp_path/'memory_text_tokens.png').is_file()
