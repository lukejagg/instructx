from pydantic import BaseModel
from tests.examples._test import Test


class PrimitiveClass(BaseModel):
    age: int


PrimitiveClassTest = Test(
    name="PrimitiveClass",
    type=PrimitiveClass,
    xml="<primitive_class><age>1</age></primitive_class>",
    expected_result=PrimitiveClass(age=1),
)


if __name__ == "__main__":
    PrimitiveClassTest.run()
