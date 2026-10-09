// 让 TPU v4 的 TensorCore 支持沿子通道的 gather（tpu.dynamic_gather，维度 0）和分段转置：替换 libtpu 0.0.49 里
// LloRegionBuilder::VpermuteSlane 的调用，改为“存进一块临时的 TC VMEM、vsetiar.raw、vld.iar”。
// 这里只通过地址调用 libtpu 自己的函数；地址是 ELF 虚拟地址，装载基址由 configure() 给出。
#include <cstddef>
#include <cstdint>

namespace {

std::uintptr_t base;

template <class Function>
Function at(std::uintptr_t address) {
    return reinterpret_cast<Function>(base + address);
}

using Value = void*;
using Builder = void*;
using Region = void*;
// StrideSpec 按值放在一个寄存器里；普通的整 TC VREG 读写是高、低 32 位各为 1。
const std::uint64_t kUnitStride = 0x100000001;
const int kS32 = 4;
// 装 IAR 用哪一种指令：3 是 vsetiar.raw，4 是 vsetiar.sublane，2 是 vsetiar.lane（LLO 的 opcode 编号）。
int set_iar_opcode = 3;

}  // namespace

extern "C" void configure(std::uintptr_t address) {
    base = address;
}

extern "C" void use_set_iar(int opcode) {
    set_iar_opcode = opcode;
}

// 分段转置（vsxpose，宽度 8）：一次提交、一次取回。模式 3 是 C++ 一侧 VxposeMode 的 Segmented B32，LLO dialect 的枚举里没有它。
Value fold(Builder builder, Value source) {
    // Vxpose(值, 模式, 宽度, 第几块, 一共几块, XLU 编号, optional<int>)；后两个参数在栈上。
    at<Value (*)(Builder, Value, int, int, int, int, std::uint64_t, std::uint64_t)>(0x197935e0)(builder, source, 3, 8, 0, 1, 0, 0);
    return at<Value (*)(Builder, int, int)>(0x19794230)(builder, 0, 0);  // Vxposeres(XLU 编号, 第几块)
}

extern "C" Value sublane_gather(Builder builder, Value source, Value indices, int) {
    // “用自己当下标去 gather 自己”没有意义，借它来表示分段转置：下标就是源，或者是对源的一次类型转换。
    if (indices == source || at<long (*)(Value, Value)>(0x197eb1e0)(indices, source) >= 0) return fold(builder, source);  // LloValue::FindOperandIndexFull
    Region region = *static_cast<Region*>(builder);
    // xla::Shape 按值返回（调用者给出存放位置）；它的大小没有去量，留足空间，用完不析构。
    alignas(16) static char shape[1024];
    const long dimensions[2] = {8, 128};
    at<void* (*)(void*, int, const long*, std::size_t)>(0xc3e80b0)(shape, kS32, dimensions, 2);  // xla::ShapeUtil::MakeShape
    Value address = at<Value (*)(Builder, void*, const char*, std::size_t)>(0x1975ac90)(builder, shape, "sublane_gather", 14);  // AllocateScopedVmem
    at<Value (*)(Builder, Value, Value, Value, std::uint64_t, Value)>(0x19792850)(builder, address, source, nullptr, kUnitStride, nullptr);  // Vst
    void* set;
    if (set_iar_opcode == 3) {
        // vsetiar.raw 装的是相对自己所在行的偏移，gather 的下标是绝对行号。
        Value row = at<Value (*)(Builder)>(0x1975faa0)(builder);  // Vslaneid
        Value offset = at<Value (*)(Builder, Value, Value)>(0x19762580)(builder, indices, row);  // VsubS32
        set = at<void* (*)(unsigned, Value, Region)>(0x1971f860)(0, offset, region);  // LloInstruction::CreateVectorSetIarRaw
    } else {
        set = at<void* (*)(int, unsigned, Value, Region)>(0x1971f680)(set_iar_opcode, 0, indices, region);  // LloInstruction::CreateVectorSetIarHelper
    }
    at<Value (*)(Region, void*, std::uint64_t)>(0x19752860)(region, set, 0);  // LloRegion::AppendInstruction
    // VldHelper 的第一个参数是 optional<unsigned>：第 32 位表示有值，低 32 位是 IAR 的编号。
    return at<Value (*)(Builder, std::uint64_t, Value, Value, std::uint64_t)>(0x1977d650)(builder, 1ull << 32, address, nullptr, kUnitStride);
}
