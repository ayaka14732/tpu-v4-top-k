// 接管 libtpu 0.0.49 里 TPU v4 的指令延迟表（xla::pufferfish::LatencyTablePufferfish::LatencyBetweenInternal，经虚表调用）：
// 先调用原函数，记下每一对指令（按 opcode）被问到的次数和回答，再按 rules 里登记的规则把回答调大。
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <utility>

namespace {

std::uintptr_t base;
// (前一条指令的 opcode, 后一条指令的 opcode) -> (次数, 原函数回答的最小值, 最大值)
std::map<std::pair<int, int>, long[3]> asked;
// 同一对 opcode 的延迟下限。
std::map<std::pair<int, int>, int> rules;
// 只在排程时生效、分配 XLU 时不生效的下限。
std::map<std::pair<int, int>, int> late_rules;
// 这次询问是不是来自给 XLU 操作分配 XLU 的那一遍（LloXluGraphOptimizer，代码在 0xf5d2cb0 到 0xf5dde60 之间）：顺着帧指针往上看返回地址。
__attribute__((noinline)) bool assigning() {
    void** frame = static_cast<void**>(__builtin_frame_address(0));
    for (int depth = 0; depth < 24 && frame != nullptr; ++depth) {
        std::uintptr_t address = reinterpret_cast<std::uintptr_t>(frame[1]) - base;
        if (address >= 0xf5d2cb0 && address < 0xf5dde60) return true;
        frame = static_cast<void**>(frame[0]);
    }
    return false;
}

int opcode(const void* value) {
    return *reinterpret_cast<const std::uint16_t*>(static_cast<const char*>(value) + 0x1a);
}

}  // namespace

extern "C" void configure(std::uintptr_t address) {
    base = address;
}

extern "C" void set_rule(int source, int destination, int cycles) {
    rules[{source, destination}] = cycles;
}

extern "C" int latency_between(void* table, const void* source, const void* destination) {
    int answer = reinterpret_cast<int (*)(void*, const void*, const void*)>(base + 0x191361f0)(table, source, destination);
    std::pair<int, int> key{opcode(source), opcode(destination)};
    long* entry = asked[key];
    if (entry[0]++ == 0 || answer < entry[1]) entry[1] = answer;
    if (answer > entry[2]) entry[2] = answer;
    auto rule = rules.find(key);
    if (rule != rules.end() && rule->second > answer) answer = rule->second;
    auto late = late_rules.find(key);
    if (late != late_rules.end() && late->second > answer && !assigning()) answer = late->second;
    return answer;
}

extern "C" void set_late_rule(int source, int destination, int cycles) {
    late_rules[{source, destination}] = cycles;
}

extern "C" void dump(const char* path) {
    std::FILE* file = std::fopen(path, "w");
    for (const auto& [key, entry] : asked) std::fprintf(file, "%d %d %ld %ld %ld\n", key.first, key.second, entry[0], entry[1], entry[2]);
    std::fclose(file);
}
