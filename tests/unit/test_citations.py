from langchain_core.documents import Document

from rag.citations import validate_citations


def test_valid_citations():
    chunks = [Document(page_content="a"), Document(page_content="b")]
    result = validate_citations("Something [1] and [2].", chunks)
    assert result["valid"] is True
    assert result["cited_numbers"] == [1, 2]
    assert result["missing_citations"] is False
    assert result["invalid_citations"] == []
    assert result["available_citations"] == [1, 2]


def test_missing_citations_marks_invalid():
    chunks = [Document(page_content="a")]
    result = validate_citations("No citation markers here.", chunks)
    assert result["valid"] is False
    assert result["missing_citations"] is True
    assert result["cited_numbers"] == []


def test_out_of_range_citation_is_invalid():
    chunks = [Document(page_content="a")]
    result = validate_citations("See [1] and [5].", chunks)
    assert result["valid"] is False
    assert result["invalid_citations"] == [5]
    assert result["available_citations"] == [1]
