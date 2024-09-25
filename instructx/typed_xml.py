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

from instructx.partials import Partial, PartialGenerator
from instructx.xml import IncrementalXMLParser, XMLChunkAction, XMLFieldPartial, XMLFieldPartialAction

print = logger.info

T = TypeVar("T")
BaseT = TypeVar("BaseT", bound=BaseModel)


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
                        self.partial_generator.update(
                            self.current_tag, field_chunk.text
                        )
                        yield self.partial_generator.main_partial.partial()
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

        yield self.partial_generator.main_partial.parse()
        return self.partial_generator.main_partial.parse()
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
            iterator = [iterator]

        type_parser = IncrementalTypeParser(model)
        for res in type_parser.iterparse(iterator):
            pass
        return res

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

