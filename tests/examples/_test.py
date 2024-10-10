from typing import Tuple, Type, TypeVar
from pydantic import BaseModel
from loguru import logger
from instructx import TypedXML

T = TypeVar("T", bound=None)


class Test(BaseModel):
    name: str
    type: Type[T]
    xml: str | list[str]
    expected_result: T | list[T]

    def run(self) -> Tuple[T, bool]:
        if isinstance(self.xml, list):
            results = []
            for xml in self.xml:
                parsed_data = TypedXML.parse(model=self.type, iterator=xml)
                assert parsed_data == self.expected_result
                results.append(parsed_data)
            return results, True
        else:
            parsed_data = TypedXML.parse(model=self.type, iterator=self.xml)
            assert parsed_data == self.expected_result
            return parsed_data, True
