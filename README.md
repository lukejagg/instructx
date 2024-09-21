# Instruct X

Instruct X is a typed language model (LLM) generation library that uses custom XML parsing to improve the quality of generated text.

> **Note:** This library does not support future annotations.






# Current Typed XML Plan

2 parses executed in parallel:
1. XML parser
2. Type parser

Type parser will be prioritized over XML parser. For example,

test: str

<test>
    ... <-- Cursor is here
</test>

If inside of a field, there is no need to send text to the XML parser.
