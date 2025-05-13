from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from graphviz import Graph
from language.context import Context


class Node(ABC):
    def __init__(self, name: str, children: List["Element"] = None, value: str = None):
        self.name = name
        self.children = children or []
        self.value = value

    @abstractmethod
    def validate(self, context: Context) -> Tuple[bool, List[str]]:
        pass

    @abstractmethod
    def to_string(self, context: Context) -> str:
        pass

    @abstractmethod
    def __eq__(self, obj) -> bool:
        pass

    @abstractmethod
    def append_to_graph(self, graph: Graph) -> int:
        pass

    def _to_string(self, level: int = 0) -> str:
        indent = "  " * level
        result = f"{indent}{self.name}"
        if self.value:
            result += f": {self.value}"
        result += "\n"

        for child in self.children:
            result += child._to_string(level + 1)
        return result
