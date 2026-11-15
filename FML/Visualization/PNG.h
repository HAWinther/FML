#ifndef FML_VISUALIZATION_PNG_H
#define FML_VISUALIZATION_PNG_H

// =====================================================================================================================
// A self-contained PNG writer (8-bit RGB or grey), no zlib needed. The pixel rows are filtered (per row the PNG filter
// None, Sub, Up or Paeth with the smallest sum of absolute values, the usual heuristic) and compressed with deflate:
// LZ77 (32 kB window, hash chains) and the fixed Huffman codes of RFC 1951. That is a few times larger than zlib at its
// best setting, but projected density fields and colour-mapped images compress well (incompressible data, e.g. noise,
// is stored uncompressed). Any PNG reader
// (browsers, Python, ImageMagick) reads the files.
//
//   FML::VISUALIZATION::write_png("image.png", width, height, rgb);   // rgb: width * height * 3 bytes, row by row
//   FML::VISUALIZATION::write_png_grey("image.png", width, height, grey);
//
// Rows are stored from the top of the image. Throws std::runtime_error if the file cannot be written.
// =====================================================================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace VISUALIZATION {
        namespace detail {

            inline uint32_t crc32(const uint8_t * data, size_t n, uint32_t crc = 0) {
                static const auto table = [] {
                    std::array<uint32_t, 256> t{};
                    for (uint32_t i = 0; i < 256; i++) {
                        uint32_t c = i;
                        for (int k = 0; k < 8; k++)
                            c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                        t[i] = c;
                    }
                    return t;
                }();
                crc = ~crc;
                for (size_t i = 0; i < n; i++)
                    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
                return ~crc;
            }

            inline uint32_t adler32(const uint8_t * data, size_t n) {
                uint32_t a = 1, b = 0;
                for (size_t i = 0; i < n; i++) {
                    a = (a + data[i]) % 65521u;
                    b = (b + a) % 65521u;
                }
                return (b << 16) | a;
            }

            /// Bits to bytes, least significant bit first (as deflate wants)
            class BitWriter {
              public:
                std::vector<uint8_t> bytes;
                void put(uint32_t value, int nbits) {
                    for (int i = 0; i < nbits; i++) {
                        if (nbit == 0)
                            bytes.push_back(0);
                        bytes.back() |= uint8_t(((value >> i) & 1u) << nbit);
                        nbit = (nbit + 1) & 7;
                    }
                }
                /// A Huffman code is sent most significant bit first
                void put_code(uint32_t code, int nbits) {
                    uint32_t reversed = 0;
                    for (int i = 0; i < nbits; i++)
                        reversed |= ((code >> i) & 1u) << (nbits - 1 - i);
                    put(reversed, nbits);
                }

              private:
                int nbit = 0;
            };

            /// The fixed literal/length code of a symbol 0-287 (RFC 1951 3.2.6)
            inline void put_literal_length_symbol(BitWriter & w, int symbol) {
                if (symbol < 144)
                    w.put_code(0x30 + symbol, 8);
                else if (symbol < 256)
                    w.put_code(0x190 + (symbol - 144), 9);
                else if (symbol < 280)
                    w.put_code(symbol - 256, 7);
                else
                    w.put_code(0xC0 + (symbol - 280), 8);
            }

            inline void put_length(BitWriter & w, int length) {
                static const int base[29] = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                             31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
                static const int extra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                              2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
                int code = 28;
                while (base[code] > length)
                    code--;
                put_literal_length_symbol(w, 257 + code);
                w.put(uint32_t(length - base[code]), extra[code]);
            }

            inline void put_distance(BitWriter & w, int distance) {
                static const int base[30] = {1,    2,    3,    4,    5,    7,     9,     13,    17,  25,
                                             33,   49,   65,   97,   129,  193,   257,   385,   513, 769,
                                             1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
                static const int extra[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                                              6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
                int code = 29;
                while (base[code] > distance)
                    code--;
                w.put_code(uint32_t(code), 5);
                w.put(uint32_t(distance - base[code]), extra[code]);
            }

            /// A zlib stream (RFC 1950) of the data: one deflate block with fixed Huffman codes and LZ77 matches
            inline std::vector<uint8_t> zlib_compress(const std::vector<uint8_t> & data) {
                constexpr int window = 32768, min_match = 3, max_match = 258, max_chain = 64;
                constexpr int hash_bits = 15;
                const size_t n = data.size();
                std::vector<int> head(size_t(1) << hash_bits, -1), prev(window, -1);
                auto hash = [&](size_t i) {
                    return ((uint32_t(data[i]) << 10) ^ (uint32_t(data[i + 1]) << 5) ^ data[i + 2]) &
                           ((1u << hash_bits) - 1);
                };
                auto insert = [&](size_t i) {
                    if (i + 2 < n) {
                        const auto h = hash(i);
                        prev[i % window] = head[h];
                        head[h] = int(i);
                    }
                };

                BitWriter w;
                w.put(1, 1); // final block
                w.put(1, 2); // fixed Huffman codes
                size_t i = 0;
                while (i < n) {
                    int best_length = 0, best_distance = 0;
                    if (i + min_match <= n) {
                        int candidate = head[hash(i)];
                        for (int chain = 0; candidate >= 0 and chain < max_chain; chain++) {
                            const size_t c = size_t(candidate);
                            if (i - c > size_t(window - 1))
                                break;
                            const int limit = int(std::min<size_t>(max_match, n - i));
                            int length = 0;
                            while (length < limit and data[c + length] == data[i + length])
                                length++;
                            if (length > best_length) {
                                best_length = length;
                                best_distance = int(i - c);
                                if (length == limit)
                                    break;
                            }
                            const int next = prev[c % window];
                            if (next >= candidate)
                                break;
                            candidate = next;
                        }
                    }
                    if (best_length >= min_match) {
                        put_length(w, best_length);
                        put_distance(w, best_distance);
                        for (int k = 0; k < best_length; k++)
                            insert(i + k);
                        i += size_t(best_length);
                    } else {
                        put_literal_length_symbol(w, data[i]);
                        insert(i);
                        i++;
                    }
                }
                put_literal_length_symbol(w, 256); // end of block

                std::vector<uint8_t> out{0x78, 0x01}; // deflate, 32 kB window, fastest compression level flag
                if (w.bytes.size() <= n + (n / 65535 + 1) * 5) {
                    out.insert(out.end(), w.bytes.begin(), w.bytes.end());
                } else {
                    // Incompressible (noise): stored blocks of at most 65535 bytes instead
                    size_t pos = 0;
                    do {
                        const size_t len = std::min<size_t>(65535, n - pos);
                        out.push_back(pos + len == n ? 1 : 0);
                        out.push_back(uint8_t(len));
                        out.push_back(uint8_t(len >> 8));
                        out.push_back(uint8_t(~len));
                        out.push_back(uint8_t(~len >> 8));
                        out.insert(out.end(), data.begin() + pos, data.begin() + pos + len);
                        pos += len;
                    } while (pos < n);
                }
                const uint32_t a = adler32(data.data(), n);
                for (int s = 24; s >= 0; s -= 8)
                    out.push_back(uint8_t(a >> s));
                return out;
            }

            inline uint8_t paeth(int a, int b, int c) {
                const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
                return uint8_t((pa <= pb and pa <= pc) ? a : (pb <= pc ? b : c));
            }

            /// The filtered scanlines (a filter byte before each row)
            inline std::vector<uint8_t> filter_rows(int width, int height, int channels, const uint8_t * pixels) {
                const size_t stride = size_t(width) * channels;
                std::vector<uint8_t> out;
                out.reserve((stride + 1) * height);
                std::vector<uint8_t> best(stride), trial(stride);
                for (int y = 0; y < height; y++) {
                    const uint8_t * row = pixels + stride * y;
                    const uint8_t * up = y > 0 ? row - stride : nullptr;
                    long best_cost = -1;
                    int best_filter = 0;
                    for (int f = 0; f < 5; f++) {
                        if (f == 3)
                            continue; // Average: rarely better than Paeth here
                        long cost = 0;
                        for (size_t i = 0; i < stride; i++) {
                            const int a = i >= size_t(channels) ? row[i - channels] : 0;
                            const int b = up ? up[i] : 0;
                            const int c = (up and i >= size_t(channels)) ? up[i - channels] : 0;
                            uint8_t v = row[i];
                            if (f == 1)
                                v = uint8_t(row[i] - a);
                            else if (f == 2)
                                v = uint8_t(row[i] - b);
                            else if (f == 4)
                                v = uint8_t(row[i] - paeth(a, b, c));
                            trial[i] = v;
                            cost += v < 128 ? v : 256 - v;
                        }
                        if (best_cost < 0 or cost < best_cost) {
                            best_cost = cost;
                            best_filter = f;
                            best.swap(trial);
                        }
                    }
                    out.push_back(uint8_t(best_filter));
                    out.insert(out.end(), best.begin(), best.end());
                }
                return out;
            }

            inline void write_png_impl(const std::string & filename,
                                       int width,
                                       int height,
                                       int channels,
                                       const uint8_t * pixels) {
                if (width <= 0 or height <= 0)
                    throw std::runtime_error("[FML::VISUALIZATION::write_png] The image is empty");
                std::ofstream fp(filename, std::ios::binary);
                if (not fp)
                    throw std::runtime_error("[FML::VISUALIZATION::write_png] Cannot open [" + filename + "]");
                auto chunk = [&](const char * type, const std::vector<uint8_t> & body) {
                    std::vector<uint8_t> buffer(type, type + 4);
                    buffer.insert(buffer.end(), body.begin(), body.end());
                    const uint32_t length = uint32_t(body.size()), crc = crc32(buffer.data(), buffer.size());
                    const uint8_t len[4] = {uint8_t(length >> 24), uint8_t(length >> 16), uint8_t(length >> 8),
                                            uint8_t(length)};
                    const uint8_t c[4] = {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc)};
                    fp.write(reinterpret_cast<const char *>(len), 4);
                    fp.write(reinterpret_cast<const char *>(buffer.data()), std::streamsize(buffer.size()));
                    fp.write(reinterpret_cast<const char *>(c), 4);
                };
                const uint8_t signature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
                fp.write(reinterpret_cast<const char *>(signature), 8);
                std::vector<uint8_t> header;
                for (uint32_t v : {uint32_t(width), uint32_t(height)})
                    for (int s = 24; s >= 0; s -= 8)
                        header.push_back(uint8_t(v >> s));
                header.push_back(8);                             // bit depth
                header.push_back(uint8_t(channels == 3 ? 2 : 0)); // colour type: RGB or grey
                header.push_back(0);                             // compression
                header.push_back(0);                             // filter method
                header.push_back(0);                             // no interlace
                chunk("IHDR", header);
                chunk("IDAT", zlib_compress(filter_rows(width, height, channels, pixels)));
                chunk("IEND", {});
                if (not fp)
                    throw std::runtime_error("[FML::VISUALIZATION::write_png] Failed to write [" + filename + "]");
            }
        } // namespace detail

        /// Write an 8-bit RGB PNG: rgb holds width * height * 3 bytes, rows from the top of the image
        inline void write_png(const std::string & filename, int width, int height, const std::vector<uint8_t> & rgb) {
            if (rgb.size() != size_t(width) * size_t(height) * 3)
                throw std::runtime_error("[FML::VISUALIZATION::write_png] Expected width * height * 3 bytes");
            detail::write_png_impl(filename, width, height, 3, rgb.data());
        }

        /// Write an 8-bit grey PNG: grey holds width * height bytes, rows from the top of the image
        inline void
        write_png_grey(const std::string & filename, int width, int height, const std::vector<uint8_t> & grey) {
            if (grey.size() != size_t(width) * size_t(height))
                throw std::runtime_error("[FML::VISUALIZATION::write_png_grey] Expected width * height bytes");
            detail::write_png_impl(filename, width, height, 1, grey.data());
        }
    } // namespace VISUALIZATION
} // namespace FML
#endif
