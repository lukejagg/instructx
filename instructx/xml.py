from abc import ABC, abstractmethod
import re
import enum
from enum import Enum
import inspect
from loguru import logger
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo
from typing import (
    Any,
    Iterator,
    List,
    TypeVar,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    override,
)

print = logger.info

T = TypeVar("T")
BaseT = TypeVar("BaseT", bound=BaseModel)


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
    def __init__(self, parser, field_name: str):
        self.parser = parser  # IncrementalTypeParser
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
