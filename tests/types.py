from loguru import logger
print = logger.info

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
    class FieldPartial:
        def __init__(self, parser: 'IncrementalTypeParser', field_name: str):
            self.parser: IncrementalTypeParser = parser
            self.field_name: str = field_name
            self.field_value: str = ""
            self.current_chunk: str = ""
        
        def process_text_before_tag(self) -> Iterator[str]:
            previous_chunk = self.current_chunk.rsplit("<", 1)[0]
            self.current_chunk = self.current_chunk[len(previous_chunk):]
            if len(previous_chunk) > 0:
                self.field_value += previous_chunk
                yield previous_chunk
        
        def add(self, text: str):
            self.current_chunk += text
            if "<" in self.current_chunk:
                # Get the text before the last < and add it to the current field value.
                yield from self.process_text_before_tag()
                
                # If there is a > in the token, then we need to check if it's a closing tag.
                if ">" in self.current_chunk:
                    if self.parser._check_tag(self.current_chunk):
                        # TODO: Abstract this to its own method (duplicate code)
                        for xml_chunk in self.parser.xml_parser.feed(self.current_chunk):
                            assert xml_chunk.action == XMLChunkAction.EXIT, f"Expected EXIT but got {xml_chunk.action}"
                            assert xml_chunk.tag == self.parser.current_tag, f"Expected tag {self.parser.current_tag} but got {xml_chunk.tag}"
                            self.parser.tag_stack.pop()
                        yield "exited " + xml_chunk.tag
                    else:
                        # This is not the associated closing tag, so we add it to the current field value.
                        # We stop at the next < so the next iteration can check for tags exactly.
                        yield from self.process_text_before_tag()
            else:
                yield text
                self.field_value += text
                self.current_chunk = ""
    
    def __init__(self, model: Type[T]):
        self.model = model
        self.xml_parser = IncrementalXMLParser(model)
        
        self.field_partial: IncrementalTypeParser.FieldPartial | None = None
        self.tag_stack: list[str] = []

    def iterparse(self, iterator: Iterator[str]) -> Iterator[XMLChunk]:
        for text in self._split_end_tags(iterator):
            if self.is_inside_field:
                # If the text could potentially be a tag, then we parse it as a tag until otherwise noted.
                # TODO: Split this check into sub-tokens
                yield from self.field_partial.add(text)
            else:
                for xml_chunk in self.xml_parser.feed(text):
                    # print(xml_chunk)
                    if xml_chunk.action == XMLChunkAction.ENTER:
                        self.tag_stack.append(xml_chunk.tag)
                        yield "entered " + xml_chunk.tag
                        # TODO: implement this
                        if self.tag_stack[-1] == "child":
                            self.field_partial = IncrementalTypeParser.FieldPartial(parser=self, field_name=self.current_tag)
                        
                    elif xml_chunk.action == XMLChunkAction.EXIT:
                        assert self.current_tag == xml_chunk.tag, f"Expected tag {self.current_tag} but got {xml_chunk.tag}"
                        yield "exited " + xml_chunk.tag
                        self.tag_stack.pop()
                        self.field_partial = None
            
            # print(ET.tostring(partial).decode())
    
    def _split_end_tags(self, iterator: Iterator[str]) -> Iterator[str]:
        """
        Used to split text into relevant chunks that can be parsed by the XML parser.
        Do not send text inside of a tag to the XML parser.
        """
        for token in iterator:
            while ">" in token:
                subtoken = token.split(">", 1)[0]
                token = token[len(subtoken) + 1:]
                yield subtoken + ">"
            if len(token) > 0:
                yield token
    
    def _check_tag(self, token: str) -> bool:
        """Check if token is the correct closing tag"""
        if token == f"</{self.current_tag}>":
            return True
        return False
    
    @property
    def current_tag(self) -> str | None:
        return self.tag_stack[-1] if len(self.tag_stack) > 0 else None
    
    @property
    def is_inside_field(self) -> bool:
        return self.field_partial is not None

class TypedXML:
    @classmethod
    def parse(self, model: Type[T], iterator: Union[str, Iterator[str]]) -> Partial[T]:
        if isinstance(iterator, str):
            iterator = iter(iterator)
        return self.iterparse(iterator)

    @classmethod
    def iterparse(self, model: Type[T], iterator: Iterator[str]) -> Iterator[Partial[T]]:
        """Parse function that takes in an iterator and returns another iterator."""
        type_parser = IncrementalTypeParser(model)
        
        """
        The XML parser is used to parse the XML into its tags character-by-character.
        When a tag is opened, and that tag is a field, then we switch to the type parser.
        
        By splitting each > in the tag, we can ensure that the XML parser will not get random parts of the value inside of the tag, (e.g. "<child>data")
        and the typed parser will change states to the XML parser correctly (e.g. "data</child>")
        """
        yield from type_parser.iterparse(iterator)


print("Starting")
xml_string = '<response> <child>data</child>   <child>more data</child>  </response>'
xml_string = [xml_string[i:i+5] for i in range(0, len(xml_string), 5)]
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
