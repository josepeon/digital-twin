"""Generate PWA icon PNGs."""
import struct, zlib

def create_icon_png(size):
    width = height = size
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            t = (x + y) / (width + height)
            r = int(88 + (163 - 88) * t)
            g = int(166 + (113 - 166) * t)
            b = int(255 + (247 - 255) * t)
            row.extend([r, g, b, 255])
        rows.append(bytes([0] + row))
    raw = b"".join(rows)

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = make_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    idat = make_chunk(b"IDAT", zlib.compress(raw))
    iend = make_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend

for size in [192, 512]:
    data = create_icon_png(size)
    path = f"src/static/icon-{size}.png"
    with open(path, "wb") as f:
        f.write(data)
    print(f"Created {path} ({len(data)} bytes)")
