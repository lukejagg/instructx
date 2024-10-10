from tests.examples import EXAMPLES


for example in EXAMPLES:
    instance, success = example.run()
    print(f"{success and 'PASS' or 'FAIL'}\t{example.name:15}\t{instance}")
