from typing import List
from pydantic import BaseModel
from tests.examples._test import Test


class ListClass(BaseModel):
    age: List[int]


ListTest = Test(
    name="List",
    type=ListClass,
    xml="<primitive_class><age><value>1</value><value>2</value></age></primitive_class>",
    expected_result=ListClass(age=[1, 2]),
)


if __name__ == "__main__":
    ListTest.run()
