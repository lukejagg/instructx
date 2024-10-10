from typing import List
from pydantic import BaseModel
from tests.examples._test import Test


class ListClass(BaseModel):
    class Value(BaseModel):
        test: int

    values: List[Value]


ListClassTest = Test(
    name="ListClass",
    type=ListClass,
    xml="""<response>
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
</response>""",
    expected_result=ListClass(
        values=[
            ListClass.Value(test=1),
            ListClass.Value(test=2),
            ListClass.Value(test=3),
        ]
    ),
)


if __name__ == "__main__":
    ListClassTest.run()
