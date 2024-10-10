from tests.examples._test import Test


PrimitiveTest = Test(
    name="Primitive",
    type=int,
    xml="<int>1</int>",
    expected_result=1,
)


if __name__ == "__main__":
    PrimitiveTest.run()
