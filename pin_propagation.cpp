// libtpu 0.0.49, x86-64: PinPrecoloring 成功改变颜色后，调用原有 MemorySpacePropagation。
// 两个 RunImpl 都通过 rdi 的隐藏返回指针返回 16 字节 StatusOr<bool>；status=1 表示成功，bool 在 +8。
// 不包含 HLO/Shape 的私有字段，不重写排序；所有分析对象由 libtpu 自己创建和销毁。
#include <atomic>
#include <cstdint>

struct Result {
    std::uintptr_t status;
    unsigned char changed;
    unsigned char padding[7];
};
static_assert(sizeof(Result) == 16);

struct Propagation {
    std::uintptr_t vtable;
    void* analysis;
};
static_assert(sizeof(Propagation) == 16);

using Run = Result* (*)(Result*, void*, void*, const void*);
static std::uintptr_t base;
static std::atomic<std::uint64_t> counters[4];

extern "C" void configure(std::uintptr_t address) { base = address; }

extern "C" void statistics(std::uint64_t* out) {
    for (int i = 0; i < 4; ++i) out[i] = counters[i].load();
}

extern "C" Result* after_pin(Result* result, void* pass, void* module, const void* threads) {
    ++counters[0];
    reinterpret_cast<Run>(base + 0x0f44cb60)(result, pass, module, threads);
    // 原 pass 失败时原样返回；没有 Pin 改色时不增加任何传播。
    if (result->status != 1 || !result->changed) return result;
    ++counters[1];
    Propagation propagation{base + 0x1e132790 + 16, nullptr};
    Result propagated{};
    reinterpret_cast<Run>(base + 0x1a213190)(&propagated, &propagation, module, threads);
    reinterpret_cast<void (*)(void*)>(base + 0x1a214600)(&propagation);  // 非 deleting destructor
    if (propagated.status != 1) {
        ++counters[3];
        *result = propagated;  // 转移错误 status 的所有权给原调用者；本地 POD 不析构它。
    } else {
        counters[2] += propagated.changed != 0;
        result->changed |= propagated.changed;
    }
    return result;
}
