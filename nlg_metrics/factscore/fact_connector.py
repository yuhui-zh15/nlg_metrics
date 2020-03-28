from typing import List

class FactConnector(object):
    def __init__(self, name):
        self.name = name
    
    def connect(self, fact: List[str]) -> str:
        raise NotImplementedError

    def connect_list(self, facts: List[List[str]]) -> List[str]:
        raise NotImplementedError


class BasicFactConnector(FactConnector):
    def __init__(self):
        super(BasicFactConnector, self).__init__(name='BasicFactConnector')
    
    def connect(self, fact: List[str]) -> str:
        return ' '.join([item for item in fact if item])

    def connect_list(self, facts: List[List[str]]) -> List[str]:
        return [self.connect(fact) for fact in facts]