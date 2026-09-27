# train.cpp - 死亡测试包装：运行 EXE 并期望其**非 0 退出**（中止/异常）
#
# 用法：cmake -DEXE=<可执行文件> [-DARGS="a;b"] [-DSKIP_CODE=77] [-DEXPECT_REGEX=<正则>] -P expect_abort.cmake
#   - 退出码 0：检查失效（未中止）→ FATAL_ERROR（测试失败）
#   - 退出码 SKIP_CODE：无设备等跳过 → 正常返回（测试通过）
#   - 其他非 0（含 Windows 异常码 0xC0000409）：符合预期 → 正常返回（测试通过）
#   - EXPECT_REGEX（可选）：stdout+stderr 合并后必须匹配（用于校验中止消息内容）
#
# 说明：CTest 的 WILL_FAIL 不会反转"异常退出"（crash）的失败判定，因此用本脚本包装。

if(NOT DEFINED EXE)
    message(FATAL_ERROR "expect_abort.cmake: 需要 -DEXE=<可执行文件>")
endif()
if(NOT DEFINED ARGS)
    set(ARGS "")
endif()
if(NOT DEFINED SKIP_CODE)
    set(SKIP_CODE "77")
endif()
if(NOT DEFINED EXPECT_REGEX)
    set(EXPECT_REGEX "")
endif()

execute_process(COMMAND "${EXE}" ${ARGS} RESULT_VARIABLE result
                OUTPUT_VARIABLE out ERROR_VARIABLE err)

if(result STREQUAL "0")
    message(FATAL_ERROR "expect_abort: ${EXE} ${ARGS} 未中止（退出码 0）——检查可能失效")
elseif(result STREQUAL "${SKIP_CODE}")
    message(STATUS "expect_abort: 跳过（退出码 ${result}）")
else()
    if(NOT EXPECT_REGEX STREQUAL "")
        string(CONCAT combined "${out}" "${err}")
        if(NOT combined MATCHES "${EXPECT_REGEX}")
            message(FATAL_ERROR "expect_abort: 输出未匹配 EXPECT_REGEX（${EXPECT_REGEX}）\n"
                                "--- 输出开始 ---\n${combined}\n--- 输出结束 ---")
        endif()
    endif()
    message(STATUS "expect_abort: 已按预期中止（退出码 ${result}）")
endif()
