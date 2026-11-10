Run from the MemAtlas repository root:

```bash
.venv/bin/python test/memq/check_memq.py
```

This offline regression suite uses native MemQ memory algorithms with local
NumPy storage and deterministic framework provider fixtures. It makes no
remote LLM or embedding requests and reads no dotenv credentials. It tests
the algorithm and adapter lifecycle rather than online model quality or a
complete benchmark agent.

Source and dependencies
-----------------------

`methods/memq/_vendor` contains 21 upstream Python files covering all native
memory build/retrieve/update strategies, MemoryStore, RL/Bellman utilities,
provider prompts/retries/circuit breaker, public configuration models, and
utility modules. The included MIT license and `UPSTREAM.json` pin revision
`9d66abf4f0252b084f088d5081c44b840842e47b` and file hashes. Runtime and tests do
not depend on `plan/`, benchmark runners, environments, agents, or datasets.
Packaging adds a final newline to `service/keyer.py`, `service/strategies.py`,
`service/__init__.py`, and the two originally empty package markers
`configs/__init__.py` and `utils/__init__.py`. All other bytes, executable
algorithm text, and prompts are unchanged.

Preflight verifies the bundled revision and SHA-256 inventory without native
provider initialization. Explicit `runtime.source_roots.memq` overrides remain
supported. A source archive without Git metadata or an inventory reports an
unverified revision; missing files, changed hashes, or known wrong revisions
fail preflight.

Create a dedicated environment and install the manifest requirements:

```bash
python3 -m venv /home/lzy/.venvs/mematlas-memq
/home/lzy/.venvs/mematlas-memq/bin/python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

requirements = json.loads(Path('methods/memq/method.yaml').read_text())['requirements']
constraints = requirements['python_constraints']
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install',
    *(package + constraints.get(package, '') for package in requirements['python']),
])
PY
/home/lzy/.venvs/mematlas-memq/bin/python -m pip check
/home/lzy/.venvs/mematlas-memq/bin/python test/memq/check_memq.py
```

Core-package requirements come from upstream `pyproject.toml`; `tqdm>=4.60.0`
and `tiktoken>=0.5.0` come from upstream `requirements.txt`.
`embedding_max_token_len=0` disables optional token truncation by default;
when enabled, native tiktoken initialization also needs its encoding data in
the local cache. Native LocalEmbedder additionally requires
sentence-transformers, but this adapter uses framework embeddings through the
native OpenAIEmbedder transport boundary. Benchmark-only packages are not
installed by this memory plugin. No shared-environment packages are changed
by the delivered commands unless a caller explicitly runs them there.

Profiles and learning boundaries
-------------------------------

`default` and `lcb` select the effective official LCB memory settings:
batch/workers 50, epochs 5, proceduralization/query/adjustment,
alpha/gamma/lambda 0.3/0.5/0.8, and similarity/Q weights 0.7/0.3.
The remaining profiles preserve their effective launcher values:

| Profile | Batch/workers | Epochs | Direct-value update | Memory token limit |
| --- | ---: | ---: | --- | ---: |
| llb | 64 | 20 | epoch | 10240 |
| bfcl | 100 | 5 | batch | 4096 |
| gpqa | 100 | 5 | template: batch; runner skips for memq | 4096 |
| dsbench | 8 | 10 | batch | 8192 |
| erqa / mmmu | 50 | 5 | batch | 4096 |

All official templates select epoch MemQ propagation. DSBench's 16384-token
limit belongs to its agent; memory reflection uses 8192. LLB keeps its
trajectory/vanilla strategies, epsilon 0 and similarity/Q weights 1/0; BFCL
uses weights 0.5/0.5; GPQA uses gamma 0.35; DSBench enables normalize_dq and
adaptive initial Q with five query candidates.

Use explicit commands when reproducing runner boundaries:

```python
await memory.command('begin_epoch', {'current_epoch': 0, 'total_epochs': 5})
await memory.command('begin_batch', {})
# Submit each completed episode with its task, trajectory, boolean success,
# and explicitly confirmed retrieved memories through memory.learn(...).
await memory.command('end_batch', {})
await memory.command('end_epoch', {})
await memory.commit()
```

Epoch indices are zero based. `set_epoch_info` accepts the same epoch payload
without beginning a new epoch. In automatic mode, a scope's full `batch_size`
triggers the native batch; `end_batch` flushes a final partial batch. Epoch
updates require `end_epoch`. `commit()` persists state and pending work without
forcing learning; `close()` does not silently execute an unfinished batch.

The adapter retains direct `update_values` and Q-max refresh before native add
when the profile runner performs them. It then preserves the configured
Bellman/flush/reset order. It also preserves runner exceptions: DSBench's
provenance map has no FIFO eviction; GPQA skips direct updates for MemQ and
its batch flush omits an immediate Q-max refresh; BFCL always propagates
MemQ at epoch end and does not apply
direct epoch updates to algorithm=memq. These behaviors are documented rather
than normalized into a different strategy.

Episodes require a nonempty task, trajectory, and an external boolean
`success`. Native success/failure reward values are retained;
`EpisodeOutcome.reward` does not replace them. Only explicitly confirmed
`MemoryEpisode.retrieved` items receive credit; candidates are not silently
reused. Image event URIs are passed to native memory construction in its
`{'url': ...}` shape; benchmark-specific image prompt assembly remains with
the caller.

Providers, logs, and snapshots
----------------------------

Native OpenAILLM.generate and OpenAIEmbedder.embed retain their prompt,
retry, circuit-breaker, token truncation, and chunk/merge paths. SDK facades
adapt only the framework text/vector transport. Circuit breakers use official
settings per scope service. The framework protocol returns final text rather
than SDK streaming responses, so the native non-streaming generate branch
is used. SDK usage/finish_reason fields remain unknown and native token counters
remain zero. The framework ledger records actual provider usage separately.

Adapter logs use `mematlas.methods.memq.adapter`; native algorithms keep
`memq.*` loggers and `[MemQ]` records. Native provider logs live under
`<artifact_dir>/memq/provider_logs/`.

Schema-2 snapshots inline native `vectors.npy`/`payloads.json` bytes and the
logical sidecar: Q-max, traces, epoch data, ordered query caches,
provenance DAG, last retrieval, and pending batches/edges. Python's
process-global `random` state is saved for native epsilon-greedy. Restore uses native
MemoryStore.load plus that sidecar, without the service loader's cache
filtering or implicit re-embedding. A schema-2 state does not require a
separate snapshot directory. Incomplete legacy schema-1 adapter states are
rejected; no migration is provided.

The typed sidecar retains mapping order, NumPy values, and date/datetime
metadata that native JSON checkpoint serialization may stringify. Unsupported
arbitrary object values fail with `TypeError`.

`commit()` also writes `<storage_path>/adapter_state.json` and native
content-addressed checkpoints under `<storage_path>/scopes/`; when
`storage_path` is omitted it defaults to `<artifact_dir>/memq`. Provider logs
continue to use the artifact directory.

Loading a state or restoring a read-only query rewinds the process-global
Python RNG, which also affects other Python components using `random` in
the same process. The framework has no isolated per-plugin RNG context;
concurrent scopes do not have independent RNG streams. NumPy RNG, provider
and circuit-breaker state, usage/logs, UUID/time sources, and physical
filesystem history are outside the snapshot. Identical stochastic replay
after learning is not promised.
