#ifndef FML_HALO_HISTORY_IO_HEADER
#define FML_HALO_HISTORY_IO_HEADER
#include <FML/HOD/HOD.h>
#include <FML/HaloHistory/HaloHistory.h>
#include <charconv>
#include <functional>
#include <iomanip>
#include <istream>
#include <locale>
#include <ostream>
#include <sstream>
namespace FML::HALO_HISTORY {
    namespace io_detail {
        struct FormatGuard {
            std::ostream& out;
            std::ios::fmtflags flags;
            std::streamsize precision;
            std::locale locale;
            explicit FormatGuard(std::ostream& stream)
                : out(stream), flags(stream.flags()), precision(stream.precision()), locale(stream.getloc()) {
                out.imbue(std::locale::classic());
                out.flags(std::ios::dec);
                out.precision(17);
            }
            ~FormatGuard() {
                try {
                    out.flags(flags);
                    out.precision(precision);
                    out.imbue(locale);
                } catch (...) {
                }
            }
        };
        inline std::string token(std::istream& input) {
            std::string value;
            if (!(input >> value))
                throw std::invalid_argument("Truncated halo catalogue");
            return value;
        }
        inline void expect(std::istream& in, const std::string& expected) {
            detail::require(token(in) == expected, "Unexpected halo catalogue field/version");
        }
        inline ID integer(const std::string& value) {
            ID result = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
            detail::require(parsed.ec == std::errc() && parsed.ptr == value.data() + value.size(),
                            "Invalid unsigned 64-bit halo/particle ID/count");
            return result;
        }
        inline ID integer(std::istream& in) { return integer(token(in)); }
        inline double real(std::istream& in) {
            const auto text = token(in);
            std::istringstream parser(text);
            parser.imbue(std::locale::classic());
            double value = 0;
            parser >> value;
            detail::require(bool(parser) && parser.peek() == std::char_traits<char>::eof() && std::isfinite(value),
                            "Invalid/nonfinite halo catalogue number");
            return value;
        }
    } // namespace io_detail
    /// Portable neutral interchange, not a claim to parse finder-specific native formats.
    inline void write_catalogue(std::ostream& output, const Snapshot& s) {
        validate(s);
        io_detail::FormatGuard guard(output);
        output << std::setprecision(17) << "FML_HALO_CATALOGUE 1\nsnapshot " << s.id << "\nscale_factor "
               << s.scale_factor << "\nboxsize " << s.boxsize << "\nmembership "
               << (s.membership == Membership::Exclusive ? "exclusive" : "inclusive") << "\nmass_definition "
               << s.metadata.mass_definition << "\nlength_unit " << s.metadata.length_unit << "\nmass_unit "
               << s.metadata.mass_unit << "\nvelocity_unit " << s.metadata.velocity_unit << "\nlength_convention "
               << s.metadata.length_convention << "\nvelocity_convention " << s.metadata.velocity_convention
               << "\nid_namespace " << s.metadata.id_namespace << "\nhalos " << s.halos.size() << '\n';
        size_t members = 0;
        for (const auto& h : s.halos) {
            detail::require(h.particles.size() <= std::numeric_limits<size_t>::max() - members,
                            "Membership count overflow");
            members += h.particles.size();
            output << h.id << ' ' << (h.host ? std::to_string(*h.host) : "-") << ' ' << h.mass << ' ' << h.radius << ' '
                   << h.vmax;
            for (double x : h.position)
                output << ' ' << x;
            for (double v : h.velocity)
                output << ' ' << v;
            output << '\n';
        }
        output << "members " << members << '\n';
        for (const auto& h : s.halos)
            for (ID p : h.particles)
                output << h.id << ' ' << p << '\n';
        output << "end\n";
        if (!output)
            throw std::runtime_error("Failed writing halo catalogue");
    }
    inline Snapshot read_catalogue(std::istream& input, size_t max_halos = 10000000,
                                   size_t max_memberships = 100000000) {
        using namespace io_detail;
        expect(input, "FML_HALO_CATALOGUE");
        expect(input, "1");
        Snapshot s;
        expect(input, "snapshot");
        s.id = integer(input);
        expect(input, "scale_factor");
        s.scale_factor = real(input);
        expect(input, "boxsize");
        s.boxsize = real(input);
        expect(input, "membership");
        auto policy = token(input);
        detail::require(policy == "exclusive" || policy == "inclusive", "Invalid membership convention");
        s.membership = policy == "exclusive" ? Membership::Exclusive : Membership::Inclusive;
        expect(input, "mass_definition");
        s.metadata.mass_definition = token(input);
        expect(input, "length_unit");
        s.metadata.length_unit = token(input);
        expect(input, "mass_unit");
        s.metadata.mass_unit = token(input);
        expect(input, "velocity_unit");
        s.metadata.velocity_unit = token(input);
        expect(input, "length_convention");
        s.metadata.length_convention = token(input);
        expect(input, "velocity_convention");
        s.metadata.velocity_convention = token(input);
        expect(input, "id_namespace");
        s.metadata.id_namespace = token(input);
        expect(input, "halos");
        ID count = integer(input);
        detail::require(count <= max_halos && count <= s.halos.max_size(), "Halo catalogue exceeds read budget");
        s.halos.resize(size_t(count));
        std::map<ID, size_t> indices;
        for (size_t i = 0; i < s.halos.size(); ++i) {
            auto& h = s.halos[i];
            h.id = integer(input);
            auto host = token(input);
            if (host != "-")
                h.host = integer(host);
            h.mass = real(input);
            h.radius = real(input);
            h.vmax = real(input);
            for (double& x : h.position)
                x = real(input);
            for (double& v : h.velocity)
                v = real(input);
            detail::require(indices.emplace(h.id, i).second, "Duplicate input halo ID");
        }
        expect(input, "members");
        count = integer(input);
        detail::require(count <= max_memberships, "Membership catalogue exceeds read budget");
        for (ID i = 0; i < count; ++i) {
            ID hid = integer(input), pid = integer(input);
            auto found = indices.find(hid);
            detail::require(found != indices.end(), "Membership row references missing halo");
            s.halos[found->second].particles.push_back(pid);
        }
        expect(input, "end");
        detail::require(!(input >> std::ws).bad() && input.peek() == std::char_traits<char>::eof(),
                        "Unexpected trailing catalogue content");
        validate(s);
        return s;
    }
    /// Units are copied unchanged; only NFW HOD satellites are available without member phase space.
    inline std::vector<FML::HOD::Halo> to_hod(const Snapshot& s, double concentration,
                                              const std::function<double(const Halo&)>& dispersion,
                                              bool include_subhalos = false) {
        validate(s);
        detail::require(concentration > 0 && std::isfinite(concentration) &&
                            s.metadata.velocity_convention == "peculiar",
                        "HOD conversion requires positive concentration and peculiar velocities");
        std::vector<FML::HOD::Halo> out;
        for (const auto& h : s.halos) {
            if (h.host && !include_subhalos)
                continue;
            FML::HOD::Halo converted;
            converted.id = h.id;
            converted.mass = h.mass;
            converted.radius = h.radius;
            converted.position = h.position;
            converted.velocity = h.velocity;
            converted.concentration = concentration;
            converted.velocity_dispersion = dispersion(h);
            detail::require(converted.velocity_dispersion >= 0 && std::isfinite(converted.velocity_dispersion),
                            "Invalid caller-supplied HOD velocity dispersion");
            out.push_back(std::move(converted));
        }
        return out;
    }
} // namespace FML::HALO_HISTORY
#endif
