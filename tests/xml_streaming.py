from loguru import logger
from pydantic import BaseModel

from instructx import TypedXML


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



print("\nStarting List[int] Example\n" + "-" * 20)

xml_string_list_int = """<root><number>10</number> <number>20</number> <number>30</number> </root>"""

xml_string_list_int = [xml_string_list_int[i : i + 5] for i in range(0, len(xml_string_list_int), 5)]
for partial in TypedXML.iterparse(list[int], xml_string_list_int):
    logger.success(partial)
    # logger.success(partial.numbers)

print("\n" + "-" * 20 + "\n" + "Done with List[int] Example")

