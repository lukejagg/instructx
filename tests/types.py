from loguru import logger

print = logger.info

import enum
import inspect
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

import re
from typing import Any, Iterator, List, TypeVar, Type, Union, get_args, get_origin, get_type_hints, override
from pydantic.fields import FieldInfo

T = TypeVar("T")
BaseT = TypeVar("BaseT", bound=BaseModel)

from abc import ABC, abstractmethod

"""
Ideas:
- Fallback Value
- Streaming
"""


class PartialType(ABC):
    current_text: str
    allow_streaming: bool
    is_primitive: bool

    def __init__(self, type_: Type, is_primitive: bool, field_info: FieldInfo = None, current_text: str = ""):
        self.type_ = type_
        # self.field_info = field_info
        self.current_text = ""
        self.is_primitive = is_primitive

    def __str__(self):
        return f"{self.type_}({self.current_text})"

    def __getattr__(self, key: str):
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    def __getitem__(self, index: int):
        raise NotImplementedError(f"__getitem__ Not implemented by {self.__class__.__name__}")

    def set(self, text: str):
        self.current_text = text

    def update(self, text: str):
        self.current_text += text

    def add_child(self, tag: str):
        raise NotImplementedError(f"Not implemented by {self.__class__.__name__}")
    
    @abstractmethod
    def parse(self):
        """
        Convert the partial to the original type.
        """
        pass
    
    @abstractmethod
    def current(self):
        """
        The current value of the partial that will be returned from the iterator.
        """
        pass
    

class PartialInt(PartialType):
    def __init__(self):
        super().__init__(int, is_primitive=True)

    @override
    def parse(self):
        return int(self.current_text)
    
    @override
    def current(self):
        return int(self.current_text)


class PartialString(PartialType):
    def __init__(self):
        super().__init__(str, is_primitive=True)
        
    @override
    def parse(self):
        return str(self.current_text)
    
    @override
    def current(self):
        return str(self.current_text)


class PartialFloat(PartialType):
    def __init__(self):
        super().__init__(float, is_primitive=True)
        
    @override
    def parse(self):
        return float(self.current_text)
    
    @override
    def current(self):
        return float(self.current_text)


class PartialBool(PartialType):
    def __init__(self):
        super().__init__(bool, is_primitive=True)
        
    @override
    def parse(self):
        return bool(self.current_text)

    @override
    def current(self):
        return bool(self.current_text)


class PartialList(PartialType):
    def __init__(self, inner_type: Type):
        super().__init__(list, is_primitive=False)
        self.inner_type = inner_type
        self.data: list[PartialType] = []

    @override
    def add_child(self, tag: str) -> 'Partial':
        self.data.append(Partial(self.inner_type))
        return self.data[-1]
    
    @override
    def parse(self):
        return [item.parse() for item in self.data]
    
    @override
    def current(self):
        return [item.current() for item in self.data]



class PartialBaseModel(PartialType):
    def __init__(self, model: Type[T]):
        super().__init__(model, is_primitive=False)
        self.model = model
        self.data: dict[str, Partial] = {}
    
    def __getattr__(self, key: str):
        return self.data.get(key)
    
    @override
    def add_child(self, tag: str) -> 'Partial':
        self.data[tag] = Partial(self.model.__annotations__[tag])
        return self.data[tag]
        
    @override
    def parse(self):
        return self.model(**{field_name: field.parse() for field_name, field in self.data.items()})
    
    @override
    def current(self):
        return Partial(self.model)


TYPES: dict[Type, Type[PartialType]] = {
    int: PartialInt,
    str: PartialString,
    float: PartialFloat,
    bool: PartialBool,
}

def create_partial_instance(type_: Type) -> PartialType:
    if get_origin(type_) is list:
        inner_type = get_args(type_)[0]
        return PartialList(inner_type)
    
    if inspect.isclass(type_):
        if issubclass(type_, BaseModel):
            return PartialBaseModel(type_)
        if type_ in TYPES:
            return TYPES[type_]()
        else:
            raise ValueError(f"Model {type_} is not a BaseModel")
    
    raise ValueError(f"Unknown type {type_}")


class Partial[T]:
    """
    Partial is used to 
    """
    def __init__(self, model: Type):
        """
        self.model.__fields__: {field_name: FieldInfo}
        self.model.__annotations__: {field_name: type}
        """
        if model is None:
            raise ValueError("Either model or partial_type must be provided")

        self.model = model
        self.partial_type = create_partial_instance(model)
    
    def __str__(self):
        return f"Partial({self.partial_type})"

    def add_child(self, tag: str) -> 'Partial':
        return self.partial_type.add_child(tag)
    
    def update(self, tag: str, value: str):
        self.children[tag].update(value)
        
    def to_model(self) -> T:
        try:
            return self.partial_type.parse()
        except ValidationError as e:
            return self.partial_type
    
    def parse(self):
        return self.partial_type.parse()
        
    def set_field(self, value: str):
        self.partial_type.set(value)

    def update_field(self, value: str):
        self.partial_type.update(value)

    @property
    def is_primitive(self) -> bool:
        return self.partial_type.is_primitive
    
    def __getattr__(self, key: str):
        return getattr(self.partial_type, key)


class PartialGenerator:
    def __init__(self, model: Type[T]):
        self.model = model
        self.partial_stack: List[Partial] = []
        self.main_partial: Partial = None
    
    def enter(self, tag: str):            
        if len(self.partial_stack) == 0:
            self.partial_stack.append(Partial(self.model))
            self.main_partial = self.partial_stack[0]
        else:
            self.partial_stack.append(self.partial_stack[-1].add_child(tag))
        
    def exit(self, tag: str, text: str = None):
        self.partial_stack.pop()
    
    def update(self, tag: str, value: str):
        # TODO: Include the Field in the Partial
        self.partial_stack[-1].update_field(value)
    
    @property
    def is_editing_primitive(self) -> bool:
        return len(self.partial_stack) > 0 and self.partial_stack[-1].is_primitive
    
    def to_model(self) -> T:
        ...

class ResponseType(BaseModel):
    count: bool


partial_test = Partial(ResponseType)
# partial_test.set_field("count", "1")
print(partial_test.to_model())


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
        self.parser = ET.XMLPullParser(events=["start", "end"])
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
            if (
                len(self.current_text.strip()) > 0
                and not self.current_text.lstrip().startswith("<")
                or is_text
            ):
                # raise Exception(f"[IncrementalXMLParser] Unexpected text: {token}")
                yield XMLChunk(
                    tag=self.current_element.tag, action=XMLChunkAction.TEXT, text=token
                )

    def _read_partial(self) -> Iterator[XMLChunk]:
        """Yield parsed elements and log when elements are entered and exited."""
        for event, elem in self.parser.read_events():
            if event == "start":  # When entering a new element
                self.current_element = elem
                self.current_text = ""
                yield XMLChunk(
                    tag=elem.tag, action=XMLChunkAction.ENTER, text=elem.text
                )
            elif event == "end":  # When exiting an element
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


"""
Field Partial is used to parse the value inside of a field. (<response><field_name>value</field_name></response>)
"""


class XMLFieldPartialAction(Enum):
    TEXT = "text"
    EXIT = "exit"


class XMLFieldPartialChunk:
    def __init__(self, action: XMLFieldPartialAction, text: str = None):
        self.action = action
        self.text = text

    def __str__(self):
        return f"XMLFieldPartialChunk({self.action}, {self.text})"


class XMLFieldPartial:
    def __init__(self, parser: "IncrementalTypeParser", field_name: str):
        self.parser: IncrementalTypeParser = parser
        self.field_name: str = field_name
        self.field_value: str = ""
        self.current_chunk: str = ""
        self.exited: bool = False

    def process_text_before_tag(self) -> Iterator[XMLFieldPartialChunk]:
        previous_chunk = self.current_chunk.rsplit("<", 1)[0]
        self.current_chunk = self.current_chunk[len(previous_chunk) :]
        if len(previous_chunk) > 0:
            self.field_value += previous_chunk
            yield XMLFieldPartialChunk(
                action=XMLFieldPartialAction.TEXT, text=previous_chunk
            )

    def add(self, text: str) -> Iterator[XMLFieldPartialChunk]:
        assert not self.exited, f'XMLFieldPartial for {self.field_name} has already exited with "{self.field_value}". Could not process "{text}"'
        self.current_chunk += text
        if "<" in self.current_chunk:
            # Get the text before the last < and add it to the current field value.
            yield from self.process_text_before_tag()

            # If there is a > in the token, then we need to check if it's a closing tag.
            if ">" in self.current_chunk:
                if self.parser._check_tag(self.current_chunk):
                    # TODO: Abstract this to its own method (duplicate code)
                    for xml_chunk in self.parser.xml_parser.feed(self.current_chunk):
                        assert (
                            xml_chunk.action == XMLChunkAction.EXIT
                        ), f"Expected EXIT but got {xml_chunk.action}"
                        assert (
                            xml_chunk.tag == self.parser.current_tag
                        ), f"Expected tag {self.parser.current_tag} but got {xml_chunk.tag}"
                        # self.parser.tag_stack.pop()
                        self.exited = True
                        yield XMLFieldPartialChunk(
                            action=XMLFieldPartialAction.EXIT, text=self.field_value
                        )
                        return
                    raise Exception(f"Failed to parse tag {self.parser.current_tag}")
                else:
                    # This is not the associated closing tag, so we add it to the current field value.
                    # We stop at the next < so the next iteration can check for tags exactly.
                    yield from self.process_text_before_tag()
        else:
            yield XMLFieldPartialChunk(action=XMLFieldPartialAction.TEXT, text=text)
            self.field_value += text
            self.current_chunk = ""


class IncrementalTypeParser:
    def __init__(self, model: Type[T]):
        self.model = model
        self.xml_parser = IncrementalXMLParser(model)

        self.partial_generator: PartialGenerator = PartialGenerator(self.model)
        self.field_partial: XMLFieldPartial | None = None
        self.tag_stack: list[str] = []

    def iterparse(self, iterator: Iterator[str]) -> Iterator[Partial]:
        for text in self._split_end_tags(iterator):
            if self.is_inside_field:
                # If the text could potentially be a tag, then we parse it as a tag until otherwise noted.
                # TODO: Split this check into sub-tokens
                for field_chunk in self.field_partial.add(text):
                    if field_chunk.action == XMLFieldPartialAction.TEXT:
                        self.partial_generator.update(self.current_tag, field_chunk.text)
                        yield self.partial_generator.main_partial
                    elif field_chunk.action == XMLFieldPartialAction.EXIT:
                        self.partial_generator.exit(self.current_tag, field_chunk.text)
                        self.pop_field_partial()
            else:
                for xml_chunk in self.xml_parser.feed(text):
                    if xml_chunk.action == XMLChunkAction.ENTER:
                        self.partial_generator.enter(xml_chunk.tag)
                        self.tag_stack.append(xml_chunk.tag)
                        # yield "entered " + xml_chunk.tag
                        if self.partial_generator.is_editing_primitive:
                            self.field_partial = XMLFieldPartial(
                                parser=self, field_name=self.current_tag
                            )

                    elif xml_chunk.action == XMLChunkAction.EXIT:
                        assert (
                            self.current_tag == xml_chunk.tag
                        ), f"Expected tag {self.current_tag} but got {xml_chunk.tag}"
                        self.partial_generator.exit(xml_chunk.tag)
                        # yield "exited " + xml_chunk.tag
                        self.tag_stack.pop()
                        self.field_partial = None

        return self.partial_generator.main_partial
            # print(ET.tostring(partial).decode())

    def _split_end_tags(self, iterator: Iterator[str]) -> Iterator[str]:
        """
        Used to split text into relevant chunks that can be parsed by the XML parser.
        Do not send text inside of a tag to the XML parser.
        """
        for token in iterator:
            while ">" in token:
                subtoken = token.split(">", 1)[0]
                token = token[len(subtoken) + 1 :]
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

    def pop_field_partial(self):
        assert self.is_inside_field, f"No field partial to pop"
        assert (
            self.current_tag == self.field_partial.field_name
        ), f"Expected tag {self.current_tag} to match field name {self.field_partial.field_name}"
        self.tag_stack.pop()
        self.field_partial = None


class TypedXML:
    @classmethod
    def parse(self, model: Type[T], iterator: Union[str, Iterator[str]]) -> Partial[T]:
        if isinstance(iterator, str):
            iterator = iter(iterator)
        return self.iterparse(iterator)

    @classmethod
    def iterparse(
        self, model: Type[T], iterator: Iterator[str]
    ) -> Iterator[Partial[T]]:
        """Parse function that takes in an iterator and returns another iterator."""
        type_parser = IncrementalTypeParser(model)

        """
        The XML parser is used to parse the XML into its tags character-by-character.
        When a tag is opened, and that tag is a field, then we switch to the type parser.
        
        By splitting each > in the tag, we can ensure that the XML parser will not get random parts of the value inside of the tag, (e.g. "<child>data")
        and the typed parser will change states to the XML parser correctly (e.g. "data</child>")
        """
        yield from type_parser.iterparse(iterator)


class Test(BaseModel):
    children: list[str]

print("\nStarting\n" + "-" * 20)
xml_string = "<Test><children> <child>data</child>   <child>more data</child>  </children></Test>"
xml_string = [xml_string[i : i + 5] for i in range(0, len(xml_string), 5)]
for partial in TypedXML.iterparse(Test, xml_string):
    logger.success(partial.children)
print("\n" + "-" * 20 + "\n" + "Done")


def xml_parser(xml: str, model: Type[T]) -> T:
    def get_next_token():
        """Gets"""

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
    Example(
        string=ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINE, model=AbstractedSimpleVariable
    ),
    Example(
        string=ABSTRACTED_SIMPLE_VARIABLE_WITH_NEW_LINES, model=AbstractedSimpleVariable
    ),
    Example(string=COMPLEX_VARIABLE, model=ComplexVariable),
    Example(string=SUPER_COMPLEX_VARIABLE, model=SuperComplexVariable),
]

for example in examples:
    parsed_data = xml_parser(example.string, example.model)
    print(parsed_data)
