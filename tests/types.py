from enum import Enum
from pydantic import BaseModel
class ResponseType(BaseModel):
    count: int


import re
from typing import Any, Iterator, List, TypeVar, Type, Union
T = TypeVar("T", bound=BaseModel)

class Partial[T]:
    def __init__(self, model: Type[T]):
        self.model = model
        self.data = {}
    
    def __str__(self):
        return f"Partial({self.data})"
    

import xml.etree.ElementTree as ET

class XMLChunkAction(Enum):
    ENTER = "enter"  # Starting to parse a tag
    EXIT = "exit"  # Finished parsing a tag
    TEXT = "text"  # Text inside an element being streamed

class XMLChunk:
    def __init__(self, tag: str, action: XMLChunkAction, text: str = None):
        self.tag = tag
        self.action = action
        self.text = text

    def __str__(self):
        return f"XMLChunk({self.tag}, {self.action}, {self.text})"

class IncrementalXMLParser:
    def __init__(self, model: Type[T]):
        self.model = model
        self.parser = ET.XMLPullParser(events=['start', 'end'])
        self.current_text = ""
        self.current_element = None
        self.complete = False
    
    def _feed(self, token, is_text: bool = False) -> Iterator[XMLChunk]:
        """Feed one token (or chunk of string) to the parser."""
        self.parser.feed(token)
        if self.current_element is not None:
            self.current_text += token
            # print(f'wrote value "{token}"')
            # TODO: Don't stream if it's inside of a <...> tag or </...> tag
            # This needs to be extended from the parent parser to pass in is_text if inside of a field.
            if len(self.current_text.strip()) > 0 and not self.current_text.lstrip().startswith("<") or is_text:
                yield XMLChunk(tag=self.current_element.tag, action=XMLChunkAction.TEXT, text=token)
        
    def _read_partial(self) -> Iterator[XMLChunk]:
        """Yield parsed elements and log when elements are entered and exited."""
        for event, elem in self.parser.read_events():
            if event == 'start':  # When entering a new element
                self.current_element = elem
                self.current_text = ""
                yield XMLChunk(tag=elem.tag, action=XMLChunkAction.ENTER, text=elem.text)
            elif event == 'end':  # When exiting an element
                self.current_element = None
                if elem.text and elem.text.strip():  # If the element has text
                    ...
                    # print(f"created <{elem.tag}>{elem.text.strip()}</{elem.tag}>")
                # print(f"exited <{elem.tag}>")
                yield XMLChunk(tag=elem.tag, action=XMLChunkAction.EXIT, text=elem.text)
                elem.clear()  # Clear element to free memory
    
    def feed(self, token: str) -> Iterator[XMLChunk]:
        yield from self._feed(token)
        yield from self._read_partial()

class IncrementalTypeParser:
    def __init__(self, model: Type[T]):
        self.model = model
        self.stack = []
    
    def _feed(self, token):
        self.stack.append(token)
    
    def _read_partial(self) -> Iterator[XMLChunk]:
        pass

class TypedXML:
    @classmethod
    def parse(self, model: Type[T], iterator: Union[str, Iterator[str]]) -> Partial[T]:
        if isinstance(iterator, str):
            iterator = iter(iterator)
        return self.iterparse(iterator)

    @classmethod
    def iterparse(self, model: Type[T], iterator: Iterator[str]) -> Iterator[Partial[T]]:
        """Parse function that takes in an iterator and returns another iterator."""
        xml_parser = IncrementalXMLParser(model)
        type_parser = IncrementalTypeParser(model)
        
        for token in iterator:
            for xml_chunk in xml_parser.feed(token):
                print(xml_chunk)
            
            
            # print(ET.tostring(partial).decode())
            

print("Starting")
xml_string = '<response> <child>data</child>   <child>more data</child>  </response>'
for partial in TypedXML.iterparse(None, xml_string):
    print(partial)
print("Done")

def xml_parser(xml: str, model: Type[T]) -> T:
    
    def get_next_token():
        """Gets """
    
    print(xml)
    partial = Partial(model)
    stack = []
    
    for line in xml.split("\n"):
        line = line.strip()
        if line.startswith("<") and not line.startswith("</"):
            tag = re.match(r"<(\w+)>", line).group(1)
            stack.append(tag)
        elif line.startswith("</"):
            tag = stack.pop()
        else:
            data[stack[-1]] = line
    
    print(data)
    print(stack)
    
    return model(**data)


"""
Unit Tests for XML parsing.
"""

SimpleVariable = int

SIMPLE_VARIABLE = """<count>1</count>"""

class AbstractedSimpleVariable(BaseModel):
    count: int

ABSTRACTED_SIMPLE_VARIABLES = """<response><count>1</count></response>"""
ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINE = """<response>
    <count>1</count>
</response>"""
ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINES = """<response>
    <count>
        1
    </count>
</response>"""

class ComplexVariable(BaseModel):
    values: List[int]

COMPLEX_VARIABLE = """<response>
    <values>
        <value>1</value>
        <value>2</value>
        <value>3</value>
    </values>
</response>"""

class SuperComplexVariable(BaseModel):
    class Value(BaseModel):
        test: int
        
    values: List[Value]

SUPER_COMPLEX_VARIABLE = """<response>
    <values>
        <value>
            <test>1</test>
        </value>
        <value>
            <test>2</test>
        </value>
        <value>
            <test>3</test>
        </value>
    </values>
</response>"""

class Example(BaseModel):
    string: str
    model: Any

examples = [
    Example(string=SIMPLE_VARIABLE, model=SimpleVariable),
    Example(string=ABSTRACTED_SIMPLE_VARIABLES, model=AbstractedSimpleVariable),
    Example(string=ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINE, model=AbstractedSimpleVariable),
    Example(string=ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINES, model=AbstractedSimpleVariable),
    Example(string=COMPLEX_VARIABLE, model=ComplexVariable),
    Example(string=SUPER_COMPLEX_VARIABLE, model=SuperComplexVariable),
]

for example in examples:
    parsed_data = xml_parser(example.string, example.model)
    print(parsed_data)
