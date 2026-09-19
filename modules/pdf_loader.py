from pypdf import PdfReader

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    total_pages = len(reader.pages)

    text = ""

    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):

        extracted = page.extract_text()

        if extracted:

            text += extracted

            page_texts.append(
                {
                    "page": page_number,
                    "text": extracted
                }
            )

    return text, total_pages, page_texts