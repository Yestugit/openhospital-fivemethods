# MemAtlas Evaluation Report

## Comparison

| Method | Implementation | Mode | Benchmark | Predictions | overall_clinical_score | diagnosis_accuracy | examination_precision | treatment_overall_score | evaluation_coverage | Memory construction time (s) | Memory learning time (s) | Mean retrieval latency (ms/request) | Mean question response time (s/question) | Memory + answer model tokens (tokens) | Sampled peak process-tree RSS (MiB) | Live memory data storage (MiB) | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memora | upstream_wrapper | standard | openhospital | 100 | 0.603149 | 0.7 | 0.587048 | 0.5224 | 1 | 20.7214 | 31598 | 7883.89 | 492.46 | unknown | 3482.99 | 36.434 | 0 |

## Run 1: memora

Benchmark: `openhospital`  
Run ID: `a205f788ad5e460b989129cdb8b7a67d`  
Predictions: 100  
Failures: 0

### Quality Metrics

| Metric | Value |
| --- | --- |
| complete_clinical_coverage | True |
| diagnosis_accuracy | 0.7 |
| evaluated_cases | 100 |
| evaluation_coverage | 1 |
| examination_precision | 0.587048 |
| examination_recall | 0.239179 |
| expected_cases | 100 |
| failed_evaluations | 0 |
| metric_counts | {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100} |
| metric_missing_counts | {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0} |
| no_examination_rate | 0.01 |
| official_examination_precision | 0.592978 |
| overall_clinical_score | 0.603149 |
| pending_evaluations | 0 |
| per_department | {'血液科': {'diagnosis_accuracy': 0.7, 'examination_precision': 0.587048, 'treatment_overall_score': 0.5224, 'treatment_safety': 0.866, 'treatment_effectiveness_alignment': 0.556, 'treatment_personalization': 0.588, 'evaluated_cases': 100, 'failed_evaluations': 0, 'pending_evaluations': 0, 'expected_cases': 100, 'evaluation_coverage': 1.0, 'metric_counts': {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100}, 'no_examination_rate': 0.01, 'examination_recall': 0.23917857142857143, 'official_examination_precision': 0.5929777777777778, 'overall_clinical_score': 0.6031493333333333, 'metric_missing_counts': {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0}, 'complete_clinical_coverage': True}} |
| treatment_effectiveness_alignment | 0.556 |
| treatment_overall_score | 0.5224 |
| treatment_personalization | 0.588 |
| treatment_safety | 0.866 |

### Efficiency and Resource Measurements

| Dimension | Metric | Unit | Value |
| --- | --- | --- | --- |
| Model usage | Agent output tokens | tokens | unknown |
| Model usage | Agent embedding requests | requests | 0 |
| Model usage | Agent LLM requests | requests | 1066 |
| Model usage | Agent input tokens | tokens | unknown |
| Model usage | Memory + answer embedding requests | requests | 26714 |
| Model usage | Judge output tokens | tokens | 6125 |
| Model usage | Judge embedding requests | requests | 0 |
| Model usage | Judge LLM requests | requests | 100 |
| Model usage | Judge input tokens | tokens | 309571 |
| Time efficiency | Total scoring time | s | 0 |
| Model usage | Memory + answer LLM requests | requests | 10988 |
| Model usage | Memory output tokens | tokens | 2027524 |
| Time efficiency | Memory construction time | s | 20.7214 |
| Model usage | Memory embedding requests | requests | 26714 |
| Time efficiency | Memory learning time | s | 31598 |
| Model usage | Memory LLM requests | requests | 9922 |
| Model usage | Memory input tokens | tokens | 17977984 |
| Space overhead | Live memory data storage | MiB | 36.434 |
| Space overhead | Sampled peak process-tree RSS | MiB | 3482.99 |
| Time efficiency | Mean question response time | s/question | 492.46 |
| Time efficiency | Mean retrieval latency | ms/request | 7883.89 |
| Model usage | All recorded model tokens including judges | tokens | unknown |
| Model usage | Memory + answer model tokens | tokens | unknown |

### Run Diagnostics

| Field | Value |
| --- | --- |
| common.add_count | 102 |
| common.add_failures | 2 |
| common.agent_cache_reported_calls | 942 |
| common.agent_cache_reported_prompt_tokens | 64511209 |
| common.agent_cache_token_coverage | partial |
| common.agent_cached_prompt_tokens | unknown |
| common.agent_calls | 1066 |
| common.agent_costed_calls | 0 |
| common.agent_embedding_completion_tokens | 0 |
| common.agent_embedding_prompt_tokens | 0 |
| common.agent_embedding_token_coverage | complete |
| common.agent_embedding_total_tokens | 0 |
| common.agent_estimated_calls | 0 |
| common.agent_failed_calls | 4 |
| common.agent_known_cache_creation_prompt_tokens | 63097553 |
| common.agent_known_cached_prompt_tokens | 563995 |
| common.agent_known_completion_tokens | 217182 |
| common.agent_known_prompt_tokens | 64701424 |
| common.agent_known_total_tokens | 64918606 |
| common.agent_llm_completion_tokens | unknown |
| common.agent_llm_prompt_tokens | unknown |
| common.agent_llm_token_coverage | partial |
| common.agent_llm_total_tokens | unknown |
| common.agent_prompt_cache_hit_rate | 0.00874259 |
| common.agent_token_coverage | partial |
| common.agent_total_tokens | unknown |
| common.agent_tracked_calls | 1066 |
| common.agent_uncosted_calls | 1066 |
| common.agent_untracked_calls | 4 |
| common.cache_reported_calls | 942 |
| common.cache_reported_prompt_tokens | 64511209 |
| common.cache_token_coverage | partial |
| common.cached_prompt_tokens | unknown |
| common.completion_tokens | unknown |
| common.costed_calls | 26714 |
| common.delete_count | 0 |
| common.delete_failures | 0 |
| common.episode_end_count | 102 |
| common.episode_end_failures | 0 |
| common.estimated_calls | 0 |
| common.failed_calls | 4 |
| common.flush_count | 9 |
| common.flush_failures | 0 |
| common.judge_cache_reported_calls | 0 |
| common.judge_cache_reported_prompt_tokens | 0 |
| common.judge_cache_token_coverage | unknown |
| common.judge_cached_prompt_tokens | unknown |
| common.judge_calls | 100 |
| common.judge_costed_calls | 0 |
| common.judge_embedding_completion_tokens | 0 |
| common.judge_embedding_prompt_tokens | 0 |
| common.judge_embedding_token_coverage | complete |
| common.judge_embedding_total_tokens | 0 |
| common.judge_estimated_calls | 0 |
| common.judge_failed_calls | 0 |
| common.judge_known_cache_creation_prompt_tokens | 0 |
| common.judge_known_cached_prompt_tokens | 0 |
| common.judge_known_completion_tokens | 6125 |
| common.judge_known_prompt_tokens | 309571 |
| common.judge_known_total_tokens | 315696 |
| common.judge_llm_completion_tokens | 6125 |
| common.judge_llm_prompt_tokens | 309571 |
| common.judge_llm_token_coverage | complete |
| common.judge_llm_total_tokens | 315696 |
| common.judge_prompt_cache_hit_rate | unknown |
| common.judge_token_coverage | complete |
| common.judge_total_tokens | 315696 |
| common.judge_tracked_calls | 100 |
| common.judge_uncosted_calls | 100 |
| common.judge_untracked_calls | 0 |
| common.known_cache_creation_prompt_tokens | 63097553 |
| common.known_cached_prompt_tokens | 563995 |
| common.known_completion_tokens | 2244706 |
| common.known_prompt_tokens | 82679408 |
| common.known_total_tokens | 84924114 |
| common.memory_cache_reported_calls | 0 |
| common.memory_cache_reported_prompt_tokens | 0 |
| common.memory_cache_token_coverage | unknown |
| common.memory_cached_prompt_tokens | unknown |
| common.memory_calls | 36636 |
| common.memory_costed_calls | 26714 |
| common.memory_embedding_completion_tokens | 0 |
| common.memory_embedding_prompt_tokens | 955375 |
| common.memory_embedding_token_coverage | complete |
| common.memory_embedding_total_tokens | 955375 |
| common.memory_estimated_calls | 0 |
| common.memory_failed_calls | 0 |
| common.memory_known_cache_creation_prompt_tokens | 0 |
| common.memory_known_cached_prompt_tokens | 0 |
| common.memory_known_completion_tokens | 2027524 |
| common.memory_known_prompt_tokens | 17977984 |
| common.memory_known_total_tokens | 20005508 |
| common.memory_llm_completion_tokens | 2027524 |
| common.memory_llm_prompt_tokens | 17022609 |
| common.memory_llm_token_coverage | complete |
| common.memory_llm_total_tokens | 19050133 |
| common.memory_prompt_cache_hit_rate | unknown |
| common.memory_token_coverage | complete |
| common.memory_total_tokens | 20005508 |
| common.memory_tracked_calls | 36636 |
| common.memory_uncosted_calls | 9922 |
| common.memory_untracked_calls | 0 |
| common.process_baseline_rss_bytes | 107610112 |
| common.process_memory_monitor_error | unknown |
| common.process_memory_sample_count | 388786 |
| common.process_memory_scope | current_process_and_descendants_excludes_external_services |
| common.process_peak_rss_bytes | 3652182016 |
| common.process_peak_rss_delta_bytes | 3544571904 |
| common.prompt_cache_hit_rate | 0.00874259 |
| common.prompt_tokens | unknown |
| common.reset_count | 0 |
| common.reset_failures | 0 |
| common.response_count | 97 |
| common.response_time_s | 47768.6 |
| common.response_timing_coverage | 0.97 |
| common.retrieval_count | 700 |
| common.retrieval_failures | 0 |
| common.retrieval_max_ms | 30359.4 |
| common.retrieval_median_ms | 7785.1 |
| common.retrieval_time_ms | 5.51872e+06 |
| common.run_wall_time_ms | 4.13348e+07 |
| common.storage_bytes | 38203772 |
| common.storage_scope | live_method_data_excludes_logs_and_recovery_snapshots |
| common.token_coverage | partial |
| common.tracked_calls | 37702 |
| common.uncosted_calls | 10988 |
| common.untracked_calls | 4 |
| common.update_count | 0 |
| common.update_failures | 0 |
| current_process_usage | {'tracked_calls': 20340, 'cached_prompt_tokens': None, 'known_cached_prompt_tokens': 236180, 'known_cache_creation_prompt_tokens': 43528606, 'cache_reported_calls': 511, 'cache_reported_prompt_tokens': 44238505, 'cache_token_coverage': 'partial', 'prompt_cache_hit_rate': 0.005338788008319902, 'llm_calls': 6349, 'embedding_calls': 13991, 'prompt_tokens': 55230642, 'completion_tokens': 1263459, 'total_tokens': None, 'known_total_tokens': 56494101, 'costed_calls': 13991, 'uncosted_calls': 6349, 'untracked_calls': 2, 'token_coverage': 'partial', 'estimated_calls': 0, 'failed_calls': 2} |
