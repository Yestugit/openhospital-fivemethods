"""HotpotQA's ExpeL sample: online Wikipedia tools, actual feedback, answer EM/F1.

Use a plain JSON array converted from the pinned upstream sample, or official
HotpotQA records (_id/question/answer). Defaults are a small 2-train/3-test split
in source order, not paper folds. No context or supporting facts reach the Agent.
For larger experiments instantiate HotpotQABenchmark with explicit counts.
"""
from collections import Counter
import json
from pathlib import Path
import re
import string

from mematlas.benchmarks.base import BenchmarkAdapter
from mematlas.core.evaluation import BenchmarkCase


def normalize_answer(value: str) -> str:
    text=''.join(c for c in value.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def answer_scores(output: str, reference: str) -> tuple[float,float]:
    actual,expected=normalize_answer(output),normalize_answer(reference)
    em=float(actual==expected)
    if actual in {'yes','no','noanswer'} or expected in {'yes','no','noanswer'}:
        if actual!=expected:return em,0.0
    a,b=actual.split(),expected.split()
    overlap=sum((Counter(a)&Counter(b)).values())
    return em, (2*overlap/(len(a)+len(b)) if a and b else em)


class HotpotQABenchmark(BenchmarkAdapter):
    benchmark_id='hotpotqa'

    def __init__(self,path: str|Path,*,train_count: int=2,test_count: int=3,max_steps: int=7):
        if any(type(n) is not int or n<1 for n in (train_count,test_count,max_steps)):
            raise ValueError('HotpotQA counts and max_steps must be positive integers')
        self.path=Path(path);self.train_count=train_count;self.test_count=test_count;self.max_steps=max_steps
        self.records=json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(self.records,list) or len(self.records)<train_count+test_count:
            raise ValueError('HotpotQA needs enough records for disjoint train/test splits')
        ids=[]
        for row in self.records:
            identity=row.get('id',row.get('_id'))
            if not isinstance(identity,str) or not identity or any(
                not isinstance(row.get(k),str) or not row[k].strip() for k in ('question','answer')):
                raise ValueError('HotpotQA records require id/_id, question and answer strings')
            ids.append(identity)
        if len(set(ids))!=len(ids):raise ValueError('Duplicate HotpotQA record identities')

    def cases(self,split='test'):
        if split not in {'train','test','all'}:raise ValueError('HotpotQA split must be train/test/all')
        for offset,row in enumerate(self.records[:self.train_count+self.test_count]):
            part='train' if offset<self.train_count else 'test'
            if split not in {part,'all'}:continue
            identity=row.get('id',row.get('_id'))
            yield BenchmarkCase(f'hotpotqa:{part}:{identity}',row['question'],reference=row['answer'],
                scope={'user_id':'expel-hotpotqa'},metadata={'benchmark_id':'hotpotqa','split':part,
                    'dataset_id':identity,'source_index':offset,'max_steps':self.max_steps})

    def native_score(self,predictions):
        scores=[answer_scores(str(p.output or ''),str(p.reference)) for p in predictions]
        return {'success_rate':sum(s[0] for s in scores)/len(scores) if scores else 0.0,
                'answer_em':sum(s[0] for s in scores)/len(scores) if scores else 0.0,
                'answer_f1':sum(s[1] for s in scores)/len(scores) if scores else 0.0,
                'question_count':len(scores)}

    def describe(self):
        return {'benchmark_id':self.benchmark_id,'path':str(self.path),'sha256':self.file_sha256(self.path),
            'environment':'wikipedia_docstore_online','train_count':self.train_count,'test_count':self.test_count,
            'max_steps':self.max_steps,'split_policy':'disjoint source order; small validation, not paper folds',
            'requirements':['answer LLM','requests','wikipedia==1.4.0','Wikipedia network access'],
            'agent_inputs':'question only; Search/Lookup/Finish; reference only for environment feedback/scoring',
            'qa':'ExpeLQA registered multi-step path; other methods require a compatible external Agent'}
