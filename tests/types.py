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
        
        # Move this logic to type_parser.
        class State(Enum):
            OPEN = "open"
            INSIDE_FIELD = "typing_value"
        
        current_state: State = State.OPEN
        current_field_value: str | None = None  # Current field being parsed.
        current_token : str = ""  # Current token being parsed, added to current_field after token is closed.
        tag_stack: list[str] = []
        
        def break_every_tag(iterator: Iterator[str]) -> Iterator[str]:
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
                    
        def check_tag(token: str) -> bool:
            nonlocal tag_stack
            """Check if token is the correct closing tag"""
            current_tag = tag_stack[-1]
            if token == f"</{current_tag}>":
                return True
            return False
        
        """
        The XML parser is used to parse the XML into its tags character-by-character.
        When a tag is opened, and that tag is a field, then we switch to the type parser.
        
        By splitting each > in the tag, we can ensure that the XML parser will not get random parts of the value inside of the tag, (e.g. "<child>data")
        and the typed parser will change states to the XML parser correctly (e.g. "data</child>")
        """
        for token in break_every_tag(iterator):
            if current_state == State.INSIDE_FIELD:
                # type_parser._feed(token)
                # If the text could potentially be a tag, then we parse it as a tag until otherwise noted.
                if "<" in token or current_token.startswith("<"):
                    current_token += token
                    # Get the text before the last < and add it to the current field value.
                    previous_token = current_token.rsplit("<", 1)[0]
                    current_token = current_token[len(previous_token):]
                    
                    if len(previous_token) > 0:
                        current_field_value += previous_token
                        yield previous_token
                    
                    if ">" in current_token:
                        if check_tag(current_token):
                            # TODO: Abstract this to its own method (duplicate code)
                            for xml_chunk in xml_parser.feed(current_token):
                                assert xml_chunk.action == XMLChunkAction.EXIT, f"Expected EXIT but got {xml_chunk.action}"
                                assert xml_chunk.tag == tag_stack[-1], f"Expected tag {tag_stack[-1]} but got {xml_chunk.tag}"
                                tag_stack.pop()
                            current_state = State.OPEN
                            yield "exited " + xml_chunk.tag
                        else:
                            # This is not the associated closing tag, so we add it to the current field value.
                            # We stop at the next < so the next iteration can check for tags exactly.
                            previous_token = current_token.rsplit("<", 1)[0]
                            current_token = current_token[len(previous_token):]
                            
                            if len(previous_token) > 0:
                                current_field_value += previous_token
                                yield previous_token
                else:
                    yield token
                    current_field_value += token
                    current_token = ""
                
                # TODO: Split this check into sub-tokens
            else:
                for xml_chunk in xml_parser.feed(token):
                    # print(xml_chunk)
                    if xml_chunk.action == XMLChunkAction.ENTER:
                        tag_stack.append(xml_chunk.tag)
                        yield "entered " + xml_chunk.tag
                        
                        # Check if the tag is inside of a field
                        if tag_stack[-1] == "child":
                            current_state = State.INSIDE_FIELD
                            current_field_value = ""
                            current_token = ""
                        
                    elif xml_chunk.action == XMLChunkAction.EXIT:
                        yield "exited " + xml_chunk.tag
                        # Verify that the tag is the same as the tag on the stack.
                        if tag_stack[-1] == xml_chunk.tag:
                            tag_stack.pop()
                        else:
                            raise ValueError(f"Expected tag {tag_stack[-1]} but got {xml_chunk.tag}")
            
            # print(ET.tostring(partial).decode())


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
