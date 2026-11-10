"""ExpeL 测评适配：补充训练结束标记，并按任务选择答题流程。

普通任务复用 DefaultQA；benchmark_id 为 hotpotqa 时使用工具环境执行多步问答。
prepare_episode 只整理数据，在当前归属的最后一份训练经历补 end_of_training=True。
"""
from dataclasses import replace
from mematlas.evaluation.qa import DefaultQA

class ExpeLQA(DefaultQA):
    async def __call__(self, case, memory):
        if case.metadata.get('benchmark_id') == 'hotpotqa':
            from .hotpotqa_qa import execute_hotpotqa
            return await execute_hotpotqa(self,case,memory)
        return await super().__call__(case,memory)

    @classmethod
    def validate_cases(cls, config, train, tests):
        from .hotpotqa_prompts import FEWSHOTS
        questions={example.split('\n')[0].removeprefix('Question: ').strip() for example in FEWSHOTS}
        for case in (*train,*tests):
            if case.metadata.get('benchmark_id')!='hotpotqa':continue
            if str(case.query).strip() in questions:
                raise ValueError('HotpotQA evaluation case overlaps upstream static examples')
            if not isinstance(case.reference,str) or not case.reference:
                raise ValueError('HotpotQA environment needs a private answer for scoring')
            if type(case.metadata.get('max_steps')) is not int or not 1<=case.metadata['max_steps']<=30:
                raise ValueError('HotpotQA max_steps must be within 1..30')

    @staticmethod
    def prepare_episode(case, episode, *, last_for_scope):
        if not last_for_scope:
            return episode
        return replace(episode, extensions={**episode.extensions, 'expel': {
            **episode.extensions.get('expel', {}), 'end_of_training': True}})
