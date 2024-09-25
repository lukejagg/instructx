from abc import ABC, abstractmethod
import enum
from enum import Enum
import inspect
from loguru import logger

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


"""
Ideas:
- Fallback Value
- Streaming
"""


class PartialType(ABC):
    current_text: str
    allow_streaming: bool
    is_primitive: bool

    def __init__(
        self,
        type_: Type,
        is_primitive: bool,
        field_info: FieldInfo = None,
    ):
        self.type_ = type_
        # self.field_info = field_info
        self.current_text = ""
        self.is_primitive = is_primitive

    def __str__(self):
        return f"{self.type_}({self.current_text})"

    def __repr__(self):
        return f"{self.type_}({self.current_text})"

    def set(self, text: str):
        self.current_text = text

    def update(self, text: str):
        self.current_text += text

    def add_child(self, tag: str):
        raise NotImplementedError(
            f"add_child Not implemented by {self.__class__.__name__}"
        )

    @abstractmethod
    def parse(self):
        """
        Convert the partial to the original type.
        """
        pass

    @abstractmethod
    def partial(self):
        """
        The current value of the partial that will be returned from the iterator.
        """
        pass

    @abstractmethod
    def done(self):
        pass


class PartialInt(PartialType):
    def __init__(self):
        super().__init__(int, is_primitive=True)
        self._value: int = None

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return int(self.current_text)

    @override
    def partial(self):
        return self.parse()

    @override
    def done(self):
        self._value = self.parse()


class PartialString(PartialType):
    def __init__(self):
        super().__init__(str, is_primitive=True)
        self._value: str = None

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return str(self.current_text)

    @override
    def partial(self):
        return self.parse()

    @override
    def done(self):
        self._value = self.parse()


class PartialFloat(PartialType):
    def __init__(self):
        super().__init__(float, is_primitive=True)

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return float(self.current_text)

    @override
    def partial(self):
        return self.parse()

    @override
    def done(self):
        self._value = self.parse()


class PartialBool(PartialType):
    def __init__(self):
        super().__init__(bool, is_primitive=True)
        self._value: bool = None

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return bool(self.current_text)

    @override
    def partial(self):
        return self.parse()

    @override
    def done(self):
        self._value = self.parse()


class PartialList(PartialType):
    def __init__(self, inner_type: Type):
        super().__init__(list, is_primitive=False)
        self.inner_type = inner_type
        self.data: list[PartialType] = []
        self._value: list = None
        self._partial: Partial[list] = Partial(list, items=self.data)

    @override
    def add_child(self, tag: str) -> "PartialType":
        self.data.append(create_partial_instance(self.inner_type))
        return self.data[-1]

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return [item.parse() for item in self.data]

    @override
    def partial(self):
        if self._value is not None:
            return self._value
        return self._partial

    @override
    def done(self):
        self._value = self.parse()


class PartialBaseModel(PartialType):
    def __init__(self, model: Type[T]):
        super().__init__(model, is_primitive=False)
        self.model = model
        self.data: dict[str, Partial] = {}
        self._value = None

    @override
    def add_child(self, tag: str) -> "Partial":
        self.data[tag] = create_partial_instance(self.model.__annotations__[tag])
        return self.data[tag]

    @override
    def parse(self):
        if self._value is not None:
            return self._value
        return self.model(
            **{field_name: field.parse() for field_name, field in self.data.items()}
        )

    @override
    def partial(self):
        if self._value is not None:
            return self._value
        return Partial(
            self.model, attrs={key: value for key, value in self.data.items()}
        )

    @override
    def done(self):
        self._value = self.parse()


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
    def __init__(
        self,
        partial_type: PartialType,
        attrs: dict[str, PartialType] = None,
        items: list[PartialType] = None,
    ):
        """
        self.model.__fields__: {field_name: FieldInfo}
        self.model.__annotations__: {field_name: type}
        """
        self.partial_type = partial_type
        self._attrs: dict[str, PartialType] = attrs
        self._items: list[PartialType] = items

    def __str__(self):
        if self._items is not None:
            items = ", ".join([repr(item.partial()) for item in self._items])
            return f"[{items}, ...]"
        if self._attrs is not None:
            attrs = " ".join(
                [f"{key}={repr(value.partial())}" for key, value in self._attrs.items()]
            )
            return f"Partial.{self.partial_type.__name__}({attrs})"
        return f"Partial.{self.partial_type.__name__}"

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, key: str):
        if not self._attrs:
            raise AttributeError(f"Partial {self.partial_type} has no attributes")
        if key not in self._attrs:
            return None
        return self._attrs[key].partial()

    def __getitem__(self, index: int):
        return self._items[index].partial()


class PartialGenerator:
    def __init__(self, model: Type[T]):
        self.model = model
        self.partial_stack: List[PartialType] = []
        self.main_partial: PartialType = None

    def enter(self, tag: str):
        if len(self.partial_stack) == 0:
            self.partial_stack.append(create_partial_instance(self.model))
            self.main_partial = self.partial_stack[0]
        else:
            self.partial_stack.append(self.partial_stack[-1].add_child(tag))

    def exit(self, tag: str, text: str = None):
        self.partial_stack.pop().done()

    def update(self, tag: str, value: str):
        # TODO: Include the Field in the Partial
        self.partial_stack[-1].update(value)

    @property
    def is_editing_primitive(self) -> bool:
        return len(self.partial_stack) > 0 and self.partial_stack[-1].is_primitive

    def to_model(self) -> T: ...
