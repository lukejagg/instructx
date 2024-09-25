from typing import List
from loguru import logger
from pydantic import BaseModel

from instructx import TypedXML

"""
Examples
"""


class Test(BaseModel):
    children: list[str]
    friends: list[list[int]]


print("\nStarting\n" + "-" * 20)
xml_string = """<Test>
<children> <child>data</child>   <child>more data</child>  </children>
<friends> 
<friend_group_1> <friend>1</friend> <friend>2</friend> </friend_group_1>
<friend_group_2> <friend>3</friend> <friend>4</friend> </friend_group_2>
</friends>
</Test>"""
xml_string = [xml_string[i : i + 5] for i in range(0, len(xml_string), 5)]
for partial in TypedXML.iterparse(Test, xml_string):
    logger.success(partial)
    logger.success(partial.children)
    logger.success(partial.friends)
print("\n" + "-" * 20 + "\n" + "Done")


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
    parsed_data = TypedXML.parse(model=example.model, iterator=example.string)
    print(parsed_data)
