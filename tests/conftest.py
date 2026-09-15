import pymupdf
import pytest


@pytest.fixture(scope="session")
def small_pdf() -> bytes:
    """Three-page PDF: title, sections with prose, a figure, and a scanned (image-only) page."""
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 90), "Quarterly Performance Review", fontsize=22)
    p.insert_text((72, 140), "1. Revenue", fontsize=16)
    y = 170
    for i in range(12):
        p.insert_text(
            (72, y),
            f"Revenue grew {i + 3}% in region {i} driven by new enterprise accounts.",
            fontsize=10,
        )
        y += 16
    p.insert_text((72, y + 20), "2. Costs", fontsize=16)
    p.insert_text(
        (72, y + 50),
        "Operating costs were flat year over year despite headcount growth.",
        fontsize=10,
    )
    p.draw_rect(
        pymupdf.Rect(72, y + 80, 372, y + 280), color=(0, 0, 1), fill=(0.8, 0.8, 1), width=2
    )
    p.insert_text((80, y + 300), "Figure 1. Cost breakdown by department.", fontsize=9)

    p2 = doc.new_page()
    p2.insert_text((72, 90), "3. Outlook", fontsize=16)
    p2.insert_text(
        (72, 120), "Guidance for next year assumes continued enterprise momentum.", fontsize=10
    )

    scan = doc.new_page()  # image-only page
    pix = doc[0].get_pixmap(dpi=50)  # `p` is stale once more pages exist
    scan.insert_image(scan.rect, pixmap=pix)
    return doc.tobytes()
