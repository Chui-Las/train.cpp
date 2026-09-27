# train.cpp - CPU 参考后端
add_library(traincpp_cpu STATIC
    trc_cpu.cpp
    trc_cpu_ops.cpp
)
target_link_libraries(traincpp_cpu PUBLIC traincpp_core)
target_include_directories(traincpp_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)
