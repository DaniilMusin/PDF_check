from pdf_io.extractor import extract_text, PDFExtractionError


def test_good_pdf(fixtures_dir):
    txt = extract_text(fixtures_dir / "sample.pdf")
    assert len(txt) > 10


def test_encrypted_pdf(fixtures_dir):
    import pytest

    with pytest.raises(PDFExtractionError):
        extract_text(fixtures_dir / "encrypted.pdf")
