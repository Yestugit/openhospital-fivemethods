# MemQ adapter

The plugin uses native MemQ revision
`9d66abf4f0252b084f088d5081c44b840842e47b` for memory construction,
NumPy storage/search, direct Q updates, Bellman TD(lambda), eligibility traces,
and provenance propagation. All upstream build/retrieve/update strategies,
provider prompts, retries, and circuit-breaker code are retained.

The complete memory import closure is included under [_vendor](_vendor), with
the original MIT [LICENSE](_vendor/LICENSE) and
[revision/file inventory](_vendor/UPSTREAM.json). Runtime and tests do not
require `plan/` or benchmark dependencies.

The default follows effective LCB settings: batch size/workers 50, five
epochs, batch direct-value updates and epoch MemQ propagation. Other official
profiles and their effective parameter differences appear in
[method.yaml](method.yaml).

Train/test evaluation now executes the configured epochs and finishes each
epoch with the native Bellman pass, including its final partial batch.
Streaming and interactive evaluation finish one pass; OpenHospital follows
the explicit batches in its experience plan. Custom training loops call
`memory.learning_boundary("epoch_start", current_epoch=0, total_epochs=1)`
and `memory.learning_boundary("epoch_end")`, followed by `memory.flush()`.

For online train/test learning, supply `training_agent` to
`ExperienceTrainTestProtocol`, or `agent` to `ExperimentRunner.evaluate`.
The agent returns an evaluated `AgentTrace` from its new execution.
Historical imports use their recorded labels and applications; they do not
credit a new retrieval with a historical trajectory's reward. Reports record
the selected training mode.

See [the reproducible offline test instructions](../../test/memq/README.md)
for manifest-based installation, batch/epoch commands, source verification,
and the test command. Objective differences and framework limitations are in
[difference.md](../../difference.md).
