"""Explicit controlled services and navigation traces; never benchmark/quality results."""
from copy import deepcopy
from mematlas import MemoryEpisode,MemoryEvent
from mematlas.core.plugin import MethodServices

WORKFLOW='## Search for an item\nFind an item by its name.\n1. Fill the search box with {item}.\n2. Click the search button.'


class Induction:
    def __init__(self):self.calls=[]
    async def complete(self,messages,**options):
        self.calls.append({'messages':deepcopy(messages),'options':options})
        return WORKFLOW


def services():return MethodServices(llm=Induction())


def config(path,profile='mind2web',**params):
    return {'method':{'method_id':'awm','params':{'profile':profile,**params}},
            'scope':{'user_id':'alice'},'runtime':{'artifact_dir':str(path)}}


def episode(identity,*,website='shop',success=True,end=False,canonical=False,template_id=1):
    return MemoryEpisode(identity,task='Find the named item',success=success,
        events=[MemoryEvent('observation','[textbox 1] Search [button 2] Go',event_id=identity+'-obs'),
                MemoryEvent('action',"fill('1', 'item')",event_id=identity+'-action',
                    extensions={'awm':{'thought':'Use the search box.'}})],
        extensions={'awm':{'website':website,'domain':'Shopping','subdomain':'General',
            'template_id':template_id,'canonical':canonical,'end_of_training':end}})
