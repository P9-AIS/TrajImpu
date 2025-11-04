from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IDataConnection(ABC):

    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None, fetch_results: bool = True) -> Optional[List[Any]]:
        raise NotImplementedError
