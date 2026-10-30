#ifndef FML_PERFORMANCE_STORAGE_H
#define FML_PERFORMANCE_STORAGE_H
#include <FML/Global/Global.h>
#include <FML/Performance/Counters.h>

namespace FML::PERFORMANCE {
    // Used only for FFTWGrid payload storage in instrumented builds. Preserve the
    // existing allocator and default grid metadata copy/move semantics.
    template <class T>
    class Storage : public FML::Vector<T> {
        using Base = FML::Vector<T>;
        void allocated() const { grid_allocation(this->capacity() * sizeof(T)); }

      public:
        Storage() = default;
        explicit Storage(size_t count) : Base(count) { allocated(); }
        Storage(const Storage & rhs) : Base(static_cast<const Base &>(rhs)) {
            allocated();
            grid_copy(rhs.size() * sizeof(T));
        }
        Storage(Storage &&) noexcept = default;
        Storage & operator=(Storage &&) noexcept = default;
        Storage & operator=(const Storage & rhs) { return operator=(static_cast<const Base &>(rhs)); }
        Storage & operator=(const Base & rhs) {
            if (static_cast<const Base *>(this) != &rhs) {
                const auto capacity = this->capacity();
                Base::operator=(rhs);
                if (this->capacity() != capacity)
                    allocated();
                grid_copy(rhs.size() * sizeof(T));
            }
            return *this;
        }
        Storage & operator=(Base && rhs) noexcept {
            Base::operator=(std::move(rhs));
            return *this;
        }
        void resize(size_t count) {
            const auto capacity = this->capacity();
            Base::resize(count);
            if (this->capacity() != capacity)
                allocated();
        }
    };
} // namespace FML::PERFORMANCE
#endif
