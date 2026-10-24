"""Generate the toy-room sample PDF used by the toy-q* cases in evals/golden_set.jsonl.

Writes a 2-page text PDF with no third-party dependencies, so pypdf extracts it cleanly.
Keep the facts below in sync with the expected answers in the golden set.

Usage:
    python scripts/make_sample_pdf.py [output_path]
"""

import sys

PAGES = [
    [
        ("F2", 18, "Toy Room Inventory - Test Document"),
        ("F1", 11, ""),
        ("F2", 13, "1. Bag A"),
        ("F1", 11, "Bag A contains 8 blue balls and 10 red balls."),
        ("F1", 11, "Bag A is made of green canvas and is kept on the top shelf."),
        ("F1", 11, "Each blue ball in Bag A weighs 50 grams. Each red ball weighs 40 grams."),
        ("F1", 11, ""),
        ("F2", 13, "2. Bag B"),
        ("F1", 11, "Bag B contains 5 green balls, 3 blue balls and 2 yellow balls."),
        ("F1", 11, "Bag B is a small leather bag kept under the desk."),
        ("F1", 11, ""),
        ("F2", 13, "3. Box C"),
        ("F1", 11, "Box C holds 12 wooden cubes and 4 toy cars. It does not contain any balls."),
        ("F1", 11, "Box C is owned by Priya. Bag A and Bag B are owned by Arjun."),
    ],
    [
        ("F2", 13, "4. Room Rules"),
        ("F1", 11, "Rule 1: Red balls may only be used outdoors."),
        ("F1", 11, "Rule 2: Yellow balls are reserved for children under 5 years old."),
        ("F1", 11, "Rule 3: The toy room is open from 9:00 AM to 5:00 PM, Monday to Friday."),
        ("F1", 11, ""),
        ("F2", 13, "5. Change Log"),
        ("F1", 11, "On 3 March 2026, 2 red balls were removed from Bag A because they were damaged."),
        ("F1", 11, "The counts in Section 1 were recorded before this removal."),
        ("F1", 11, ""),
        ("F2", 13, "6. Contact"),
        ("F1", 11, "Questions about the toy room should be sent to the room supervisor, Meena Rao."),
    ],
]


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def content_stream(lines):
    out, y = ["BT"], 790
    for font, size, text in lines:
        if text:
            out.append(f"/{font} {size} Tf 1 0 0 1 60 {y} Tm ({esc(text)}) Tj")
        y -= size + 9
    out.append("ET")
    return "\n".join(out).encode("latin-1")


def build(path):
    objs = {}
    n_pages = len(PAGES)
    # 1 catalog, 2 pages, 3 F1, 4 F2, then (page, content) pairs
    page_ids = [5 + 2 * i for i in range(n_pages)]
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    objs[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objs[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    for pid, lines in zip(page_ids, PAGES):
        stream = content_stream(lines)
        objs[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {pid + 1} 0 R >>"
        ).encode()
        objs[pid + 1] = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
    objs[99] = b"<< /Title (Toy Room Inventory - RAG Test) /Author (RAG test fixture) >>"

    ids = sorted(objs)
    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for i in ids:
        offsets[i] = len(buf)
        buf += f"{i} 0 obj\n".encode() + objs[i] + b"\nendobj\n"
    size = max(ids) + 1
    xref = len(buf)
    buf += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for i in range(1, size):
        buf += (f"{offsets[i]:010d} 00000 n \n" if i in offsets else "0000000000 65535 f \n").encode()
    buf += f"trailer\n<< /Size {size} /Root 1 0 R /Info 99 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(buf)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/sample/toy-room-inventory.pdf")
