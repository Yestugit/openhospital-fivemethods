"""ExpeL QAEnv / LangChain 0.0.181 DocstoreExplorer compatible environment.

Wikipedia 1.4.0 runs in an isolated module namespace: HTTPS, explicit User-Agent,
bounded requests and response snapshots replace upstream unbounded network retry.
No dataset answer/supporting facts are sent to the Wikipedia backend.
"""
from __future__ import annotations
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from datetime import datetime,timezone

from .hotpotqa import normalize_answer


class WikipediaDocstore:
    def __init__(self,cache_dir: Path|None=None,*,timeout: float=15):
        self.cache_dir=Path(cache_dir) if cache_dir else None
        self.timeout=timeout
        self.records=[]

    def _search(self,term: str):
        import wikipedia.wikipedia as original
        import requests
        spec=importlib.util.spec_from_file_location('wikipedia._mematlas_isolated',original.__file__)
        wiki=importlib.util.module_from_spec(spec);spec.loader.exec_module(wiki)
        session=requests.Session()
        requests_used=0
        def request(params):
            nonlocal requests_used
            requests_used+=1
            if requests_used>12:raise RuntimeError('Wikipedia per-search request budget exceeded')
            params={**params,'format':'json'};params.setdefault('action','query')
            identity=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
            path=self.cache_dir/f'{identity}.json' if self.cache_dir else None
            if path and path.exists():
                record=json.loads(path.read_text(encoding='utf-8'))
            else:
                response=session.get('https://en.wikipedia.org/w/api.php',params=params,
                    headers={'User-Agent':'MemAtlas-ExpeL-Integration/0.1 (research; https://github.com/LeapLabTHU/ExpeL)'},
                    timeout=self.timeout)
                response.raise_for_status();data=response.json()
                if 'error' in data:raise RuntimeError(f'Wikipedia API error: {data["error"].get("code")}')
                record={'url':response.url,'fetched_at':datetime.now(timezone.utc).isoformat(),'response':data}
                if path:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
            self.records.append({'query':dict(params),'snapshot':identity,'url':record['url']})
            return record['response']
        wiki._wiki_request=request
        try:
            try:
                page=wiki.page(term)
                return page.content
            except (wiki.PageError,wiki.DisambiguationError):
                return {'not_found':f'Could not find [{term}]. Similar: {wiki.search(term)}'}
        finally:session.close()

    async def search(self,term: str):
        return await asyncio.to_thread(self._search,term)


class HotpotQAEnvironment:
    def __init__(self,answer: str,backend,*,max_steps: int=7):
        self._answer=answer;self.backend=backend;self.max_steps=max_steps
        self.curr_step=1;self.output='';self.terminated=False;self.success=False;self.truncated=False
        self.document=None;self.lookup_str='';self.lookup_index=0

    async def step(self,action: str) -> str:
        if self.terminated or self.truncated:raise ValueError('HotpotQA environment is finished')
        match=re.fullmatch(r'(\w+)\[(.+)\]',action.strip())
        kind,arg=match.groups() if match else (None,None)
        if kind=='Finish':
            self.output=arg;self.success=normalize_answer(arg)==normalize_answer(self._answer)
            self.terminated=True
            observation='Answer is CORRECT' if self.success else 'Answer is INCORRECT'
        elif kind=='Search':
            result=await self.backend.search(arg)
            self.document=None if isinstance(result,dict) else result
            observation=result['not_found'] if isinstance(result,dict) else self.document.split('\n\n')[0]
        elif kind=='Lookup':
            if self.document is None:
                observation='The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
            else:
                if arg.lower()!=self.lookup_str:self.lookup_str=arg.lower();self.lookup_index=0
                else:self.lookup_index+=1
                matches=[p for p in self.document.split('\n\n') if self.lookup_str in p.lower()]
                observation=('No Results' if not matches else 'No More Results' if self.lookup_index>=len(matches)
                    else f'(Result {self.lookup_index+1}/{len(matches)}) {matches[self.lookup_index]}')
        else:observation='Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].'
        self.curr_step+=1;self.truncated=self.curr_step>self.max_steps
        return observation.strip()
