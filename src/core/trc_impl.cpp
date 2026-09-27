// train.cpp - 错误处理与日志
#include "trc_impl.h"

#include <cstdarg>

namespace traincpp {

void abort_impl(const char* file, int line, const char* fmt, ...) {
    std::fflush(stdout);
    std::fprintf(stderr, "[traincpp][FATAL] %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    std::abort();
}

void log_impl(int level, const char* fmt, ...) {
    static const char* const names[] = {"", "DEBUG", "INFO", "WARN", "ERROR"};
    if (level < 1 || level > 4) {
        level = 2;
    }
    std::fprintf(stderr, "[traincpp][%s] ", names[level]);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

} // namespace traincpp
