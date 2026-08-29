import pytest
from extractor import extract_name, extract_address

def test_extract_name_basic():
    lines = [
        {"text": "Government of India"},
        {"text": "Some random noise"},
        {"text": "John Doe"},
        {"text": "DOB: 01/01/1990"},
        {"text": "Male"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_name(lines_text) == "John Doe"

def test_extract_name_with_hindi():
    lines = [
        {"text": "भारत सरकार"},
        {"text": "Government of India"},
        {"text": "अमित कुमार"},
        {"text": "Amit Kumar"},
        {"text": "जन्म तिथि / DOB: 15/08/1985"},
        {"text": "Male"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_name(lines_text) == "Amit Kumar"

def test_extract_name_skip_stoplist():
    lines = [
        {"text": "आधार"},
        {"text": "Aadhar"},
        {"text": "Unique Identification Authority of India"},
        {"text": "Sita Ram"},
        {"text": "DOB 12-12-2000"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_name(lines_text) == "Sita Ram"

def test_extract_name_skip_numbers():
    lines = [
        {"text": "Government of India"},
        {"text": "1234 5678"}, # Should skip because it contains digits
        {"text": "Valid Name"},
        {"text": "DOB 01/01/1990"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_name(lines_text) == "Valid Name"

def test_extract_address_basic():
    lines = [
        {"text": "Address: S/O John Smith,"},
        {"text": "123 Main Street,"},
        {"text": "Mumbai, Maharashtra, 400001"},
        {"text": "1234 5678 9012"} # Aadhaar number should stop extraction
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_address(lines_text) == "S/O John Smith, 123 Main Street, Mumbai, Maharashtra, 400001"

def test_extract_address_hindi():
    lines = [
        {"text": "पता: 45, एमजी रोड,"},
        {"text": "बेंगलुरु, कर्नाटक - 560001"},
        {"text": "VID: 1234 5678 9012 3456"} # VID should stop extraction
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_address(lines_text) == "45, एमजी रोड, बेंगलुरु, कर्नाटक - 560001"

def test_extract_address_no_anchor():
    lines = [
        {"text": "Some text"},
        {"text": "More text"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_address(lines_text) is None

def test_extract_name_no_dob():
    lines = [
        {"text": "Government of India"},
        {"text": "John Doe"}
    ]
    lines_text = [l["text"] for l in lines]
    assert extract_name(lines_text) is None
