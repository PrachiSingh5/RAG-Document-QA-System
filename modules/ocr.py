import platform
import pymupdf as fitz
import pytesseract
from PIL import Image
import io


# On Windows, Tesseract isn't automatically found on PATH unless it
# was added manually, so point directly at the known install location.
# On Linux (e.g. Streamlit Cloud), Tesseract is installed via
# packages.txt and is already discoverable on PATH, so no override
# is needed there.
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


def extract_text_with_ocr(pdf_bytes, zoom=1.5, timeout=30):
    """
    Run OCR on every page of a PDF (given as raw bytes).

    zoom: render scale factor. Higher = sharper image = better OCR
    accuracy, but slower.

    timeout: max seconds pytesseract will wait per page before
    giving up, so a bad image can't hang the app forever.

    Returns a list of {"page": page_number, "text": extracted_text}
    dicts, matching extract_pdf_text()'s output shape.
    """

    page_texts = []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    matrix = fitz.Matrix(zoom, zoom)

    try:
        for page_number in range(len(doc)):

            page = doc[page_number]
            pix = page.get_pixmap(matrix=matrix)
            image = Image.open(io.BytesIO(pix.tobytes("png")))

            try:
                text = pytesseract.image_to_string(image, timeout=timeout)
            except RuntimeError:
                continue

            if text.strip():
                page_texts.append(
                    {"page": page_number + 1, "text": text}
                )

    finally:
        doc.close()

    return page_texts


def render_page_image(pdf_bytes, page_number, zoom=1.5):
    """
    Render a single PDF page (1-indexed) to PNG bytes.

    Used to feed a page image into a vision model for diagram/image
    understanding, separately from the OCR text-extraction path.
    """

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        page = doc[page_number - 1]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        return pix.tobytes("png")

    finally:
        doc.close()