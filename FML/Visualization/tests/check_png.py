#!/usr/bin/env python3
"""Reads the files test_visualization wrote with the Python standard library only (zlib for the PNGs) and checks them:
chunk CRCs, the zlib stream, the filters, and the pixels against the raw bytes; the orientation; the .npy file."""
import struct, sys, zlib

failed = 0
def check(ok, label):
    global failed
    print(("[PASS] " if ok else "[FAIL] ") + label)
    failed += not ok

def read_png(name):
    data = open(name, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "signature"
    pos, chunks = 8, []
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        crc, = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])
        assert zlib.crc32(kind + body) == crc, "CRC of " + kind.decode()
        chunks.append((kind, body))
        pos += 12 + n
    assert chunks[0][0] == b"IHDR" and chunks[-1][0] == b"IEND"
    w, h, depth, colour = struct.unpack(">IIBB", chunks[0][1][:10])
    ch = {2: 3, 0: 1}[colour]
    raw = zlib.decompress(b"".join(b for k, b in chunks if k == b"IDAT"))
    stride, out, prev = w * ch, bytearray(), bytearray(w * ch)
    for y in range(h):
        f, row = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = row[i - ch] if i >= ch else 0
            b, c = prev[i], prev[i - ch] if i >= ch else 0
            if f == 1: row[i] = (row[i] + a) & 255
            elif f == 2: row[i] = (row[i] + b) & 255
            elif f == 3: row[i] = (row[i] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out += row
        prev = row
    return w, h, ch, bytes(out)

for name, raw in [("test_pattern.png", "test_pattern.raw"), ("test_grey.png", "test_grey.raw")]:
    try:
        w, h, ch, pixels = read_png(name)
        check(pixels == open(raw, "rb").read(), f"{name}: {w}x{h}, {ch} channel(s), valid CRCs/zlib, pixels exact")
    except Exception as e:
        check(False, f"{name}: {e}")

w, h, ch, pixels = read_png("test_orientation.png")
px = lambda x, y: pixels[(y * w + x) * 3]
check((w, h) == (40, 20) and px(0, 0) == 255 and px(1, 0) == 6 and px(0, 1) == 0 and px(39, 19) == 249,
      "orientation: iy = ny-1 is the top row, ix = 0 the left column")

data = open("test_orientation.npy", "rb").read()
n, = struct.unpack("<H", data[8:10])
header = data[10:10 + n].decode()
values = struct.unpack("<800d", data[10 + n:])
check(data[:6] == b"\x93NUMPY" and (10 + n) % 64 == 0 and "(20, 40)" in header and "'<f8'" in header and
      values[1] == 2.0 and values[19 * 40] == 1001.0 and len(data) == 10 + n + 800 * 8,
      ".npy: header, shape (ny, nx), row 0 = the bottom row")
sys.exit(1 if failed else 0)
