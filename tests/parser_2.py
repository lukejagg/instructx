import xml.etree.ElementTree as ET

class IncrementalXMLParser:
    def __init__(self):
        self.parser = ET.XMLPullParser(events=['start', 'end'])
        self.complete = False
    
    def feed(self, token):
        """Feed one token (or chunk of string) to the parser."""
        self.parser.feed(token)
        
    def read_partial(self):
        """Yield parsed elements as they are completed."""
        for event, elem in self.parser.read_events():
            if event == 'end':  # When an element is fully parsed
                yield elem
                elem.clear()  # Clear element to free memory
    
    def finalize(self):
        """Finalize the parsing process and ensure all elements are read."""
        if not self.complete:
            self.parser.close()  # Signal the end of input to finalize parsing
            self.complete = True

# Example usage
parser = IncrementalXMLParser()

# Simulate feeding the XML one token at a time (you can split however you like)
xml_string = '<root><child1><child>data</child><child>more data</child></child1></root>'
for token in xml_string:
    parser.feed(token)  # Feed one character at a time (or a token, line, etc.)

    # Check for partial parsed elements
    for partial in parser.read_partial():
        print(ET.tostring(partial).decode())

# Finalize the parser to ensure the whole XML is processed
parser.finalize()

# Check for any remaining parsed elements after finalization
for partial in parser.read_partial():
    print(ET.tostring(partial).decode())
