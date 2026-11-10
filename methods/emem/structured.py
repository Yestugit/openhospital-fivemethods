"""Opt-in JSON transport compatibility for the pinned native EMem prompts.

Selection, extraction, graph construction and ranking remain upstream code.
This boundary validates the upstream schemas before its permissive fallbacks.
"""
from copy import deepcopy
import difflib
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from mematlas.core.errors import ConfigurationError
from mematlas.providers.requests import LLMRequestPaused

PROFILE = 'json_unicode_v1'
INSTRUCTION = (
    'Return only a valid JSON object matching the original required JSON schema, without Markdown fences or commentary. '
    'Write Chinese and other Unicode characters directly, not as \\uXXXX escapes. '
    'Escape embedded quotation marks and backslashes correctly. After JSON decoding, copied source text must remain exact. '
    'Preserve the original extraction and selection criteria; do not invent facts or shorten source content.'
)
SCHEMAS = (
    ('conversation_edu_extraction_v1', 'ConversationEDUExtractionV1'),
    ('unified_event_extraction', 'UnifiedEventExtraction'),
    ('ner_query', 'QueryNerExtraction'),
    ('er_query', 'QueryERExtraction'),
    ('edu_filter_v1', 'FilteredEDUs'),
    ('argument_filter_v1', 'FilteredArguments'),
)


class JSONTransport:
    def __init__(self, journal):
        self.journal = Path(journal)
        self.lock = RLock()

    def prepare(self, kwargs):
        request = deepcopy(kwargs)
        messages = request['messages']
        response_format = request.get('response_format')
        if isinstance(response_format, type) and hasattr(response_format, 'model_validate_json'):
            schema = response_format
        else:
            system = '\n'.join(m['content'] for m in messages if m['role'] == 'system' and isinstance(m['content'], str))
            matches = [(module, name) for module, name in SCHEMAS
                       if f"'title': '{name}'" in system or f'"title": "{name}"' in system]
            if len(matches) != 1:
                raise ConfigurationError('EMem JSON compatibility requires an identified upstream output schema')
            module, name = matches[0]
            schema = getattr(importlib.import_module('emem.prompts.templates.' + module), name)
        candidates = None
        if schema.__name__ == 'FilteredEDUs':
            user = next(m for m in reversed(messages) if m['role'] == 'user')
            prefix, separator, raw = user['content'].partition('\n\nCandidate EDUs:\n')
            if not separator:
                raise ConfigurationError('Unrecognized native EMem candidate prompt')
            candidates = json.loads(raw)
            if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
                raise ConfigurationError('Invalid native EMem candidate list')
            # Change JSON wire spelling only; decoded candidates are identical.
            user['content'] = prefix + separator + json.dumps(candidates, ensure_ascii=False, indent=2)
        request['response_format'] = {'type': 'json_object'}
        messages.append({'role': 'system', 'content': INSTRUCTION})
        return request, schema, candidates

    @staticmethod
    def validate(response, schema, candidates):
        choice = response.choices[0]
        if choice.finish_reason != 'stop':
            raise ValueError('Incomplete JSON generation: ' + str(choice.finish_reason))
        value = schema.model_validate_json(choice.message.content, strict=True)
        if candidates is not None:
            # Use exactly the pinned native matching rule. Its retriever maps
            # matches back to original candidate objects, retaining source text.
            # Requiring literal copies here would reject outputs native EMem
            # legitimately accepts and introduce a different selection policy.
            if any(not difflib.get_close_matches(item, candidates, n=1, cutoff=0.85)
                   for item in value.selected_edus):
                raise ValueError('Selected EDU cannot map to an original candidate under native matching')
        return value

    def record(self, identity, attempt, request, response, error):
        from mematlas.core.telemetry import current_usage_context
        row = {'timestamp': datetime.now(timezone.utc).isoformat(), 'profile': PROFILE,
               'logical_request_id': identity, 'json_attempt': attempt + 1,
               'schema': request.get('_schema_name'), 'context': current_usage_context(),
               'request': {k:v for k,v in request.items() if k != '_schema_name'},
               'response': response.model_dump(mode='json'), 'validation_error': error}
        with self.lock:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
            with self.journal.open('a', encoding='utf8') as stream:
                json.dump(row, stream, ensure_ascii=False, default=str)
                stream.write('\n')

    async def invoke(self, request, send):
        params, schema, candidates = self.prepare(request)
        identity = uuid4().hex
        original_budget = params.get('max_tokens', params.get('max_completion_tokens', 8192))
        for attempt in range(4):
            response = await send(params)
            try:
                self.validate(response, schema, candidates)
                error = None
            except (ValueError, TypeError) as exc:
                error = str(exc)
            self.record(identity, attempt, {**params, '_schema_name': schema.__name__}, response, error)
            if error is None:
                return response
            if attempt == 3:
                raise LLMRequestPaused('EMem JSON output invalid after 3 repairs; responses and usage retained')
            params['messages'].extend([
                {'role': 'assistant', 'content': response.choices[0].message.content or ''},
                {'role': 'user', 'content': 'Repair the preceding output format using the same original input and schema. '
                 + INSTRUCTION + '\nValidation error: ' + error[:2000]},
            ])
            if response.choices[0].finish_reason == 'length':
                key = 'max_tokens' if 'max_tokens' in params else 'max_completion_tokens'
                params[key] = min(32768, max(original_budget, params.get(key, original_budget)) * 2)
