# train.cpp - 测试
add_library(traincpp_test_main STATIC test_main.cpp)
target_link_libraries(traincpp_test_main PUBLIC traincpp)
# 测试代码使用 getenv/fopen 等：关闭 MSVC 安全告警（必须在命令行定义，头文件里定义来不及）
target_compile_definitions(traincpp_test_main PUBLIC _CRT_SECURE_NO_WARNINGS)
target_include_directories(traincpp_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_core test_core.cpp)
target_link_libraries(test_core PRIVATE traincpp_test_main)
add_test(NAME core COMMAND test_core)

add_executable(test_ops test_ops.cpp)
target_link_libraries(test_ops PRIVATE traincpp_test_main)
add_test(NAME ops COMMAND test_ops)

# 分配生命周期（M2.1d / R3）：buffer_free 后张量 data/buffer 失效与重建
add_executable(test_alloc test_alloc.cpp)
target_link_libraries(test_alloc PRIVATE traincpp_test_main)
add_test(NAME alloc COMMAND test_alloc)

# autograd 构建期守卫死亡测试（M2.1e / R4+R15）：CTest 经 expect_abort.cmake 判定——
# 中止（非 0 退出）= 通过；未中止 = 失败
add_executable(test_grad_guard test_grad_guard.cpp)
target_link_libraries(test_grad_guard PRIVATE traincpp)
add_test(NAME grad_repeat_expand_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_grad_guard> -DARGS=repeat
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
add_test(NAME grad_shared_param_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_grad_guard> -DARGS=shared
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

# 图级分配覆盖校验死亡测试（M2.2b / R10）：图未覆盖 ctx 张量必须中止
add_executable(test_graph_alloc_guard test_graph_alloc_guard.cpp)
target_link_libraries(test_graph_alloc_guard PRIVATE traincpp)
add_test(NAME graph_alloc_coverage_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_graph_alloc_guard>
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

# mul_mat 形状守卫死亡测试（M2.3a）：违反 ggml_can_mul_mat 必须可读中止
add_executable(test_mul_mat_shape_guard test_mul_mat_shape_guard.cpp)
target_link_libraries(test_mul_mat_shape_guard PRIVATE traincpp)
add_test(NAME mul_mat_inner_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_mul_mat_shape_guard> -DARGS=inner
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
add_test(NAME mul_mat_batch2_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_mul_mat_shape_guard> -DARGS=batch2
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
add_test(NAME mul_mat_batch3_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_mul_mat_shape_guard> -DARGS=batch3
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

# autograd：有限差分 gradcheck + 梯度累加（M1.3）
add_executable(test_grad test_grad.cpp)
target_link_libraries(test_grad PRIVATE traincpp_test_main)
add_test(NAME grad COMMAND test_grad)

# nn 层系统（M1.5b；黄金数据缺失时自动跳过）
add_executable(test_nn test_nn.cpp)
target_link_libraries(test_nn PRIVATE traincpp_test_main)
target_compile_definitions(test_nn PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME nn COMMAND test_nn)

# 损失函数（M1.5c；黄金数据缺失时自动跳过）
add_executable(test_loss test_loss.cpp)
target_link_libraries(test_loss PRIVATE traincpp_test_main)
target_compile_definitions(test_loss PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME loss COMMAND test_loss)

# 规模压力套件（M2.2d）：默认小档（1700 万元素，核显可跑）；
# 大档（6400 万/2.5 亿）设 TRC_STRESS_SCALE=full，需大显存（Arc 测试机）
add_executable(test_stress test_stress.cpp)
target_link_libraries(test_stress PRIVATE traincpp_test_main)
add_test(NAME stress COMMAND test_stress)

# 逐 epoch 训练轨迹 PyTorch 对拍（M2.0d；黄金数据缺失时自动跳过）
add_executable(test_train_trace test_train_trace.cpp)
target_link_libraries(test_train_trace PRIVATE traincpp_test_main)
target_compile_definitions(test_train_trace PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME train_trace COMMAND test_train_trace)

# 优化器与学习率调度（M1.5a；黄金数据缺失时自动跳过）
add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim PRIVATE traincpp_test_main)
target_compile_definitions(test_optim PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME optim COMMAND test_optim)

# GGUF 读写（M1.6a；无外部数据依赖）
add_executable(test_gguf test_gguf.cpp)
target_link_libraries(test_gguf PRIVATE traincpp_test_main)
add_test(NAME gguf COMMAND test_gguf)

# checkpoint 断点续训（M1.6b；无外部数据依赖）
add_executable(test_checkpoint test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE traincpp_test_main)
add_test(NAME checkpoint COMMAND test_checkpoint)

# checkpoint 保存期守卫死亡测试（M2.3f / schema v2）：分区名重复、跨分区张量名重复必须中止
add_executable(test_checkpoint_guard test_checkpoint_guard.cpp)
target_link_libraries(test_checkpoint_guard PRIVATE traincpp)
add_test(NAME checkpoint_dup_name_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_checkpoint_guard> -DARGS=dup_name
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
add_test(NAME checkpoint_dup_tensor_abort
         COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_checkpoint_guard> -DARGS=dup_tensor
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

# 双图双优化器 + 通用 GAN（M2.3e）：冻结/解冻 + detach 输入
add_executable(test_gan test_gan.cpp)
target_link_libraries(test_gan PRIVATE traincpp_test_main)
add_test(NAME gan COMMAND test_gan)

# 二期验收（M2.4）：通用双网络对抗训练端到端（ConvTranspose1d + 多周期判别器 +
# weight_norm + grouped conv + reflect pad + 双优化器 + 断点续训 + GGUF 导出）
add_executable(test_acceptance test_acceptance.cpp)
target_link_libraries(test_acceptance PRIVATE traincpp_test_main)
add_test(NAME acceptance COMMAND test_acceptance)

# PyTorch 黄金数据对拍（数据由 scripts/gen_golden.py 生成；缺失时测试自动跳过）
add_executable(test_golden test_golden.cpp)
target_link_libraries(test_golden PRIVATE traincpp_test_main)
target_compile_definitions(test_golden PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME golden COMMAND test_golden)

# PyTorch 反向梯度对拍（M1.3c；同一份黄金数据目录，缺失时自动跳过）
add_executable(test_grad_golden test_grad_golden.cpp)
target_link_libraries(test_grad_golden PRIVATE traincpp_test_main)
target_compile_definitions(test_grad_golden PRIVATE TRC_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden")
add_test(NAME grad_golden COMMAND test_grad_golden)

# Vulkan 后端 smoke（M1.4a；无 Vulkan 设备时测试内自动跳过）
if(TRC_VULKAN)
    add_executable(test_vulkan test_vulkan.cpp)
    target_link_libraries(test_vulkan PRIVATE traincpp_test_main)
    add_test(NAME vulkan COMMAND test_vulkan)

    # 后端 conformance：CPU vs Vulkan 同图对照（M1.4b；无设备时自动跳过）
    add_executable(test_backend_conformance test_backend_conformance.cpp)
    target_link_libraries(test_backend_conformance PRIVATE traincpp_test_main)
    add_test(NAME backend_conformance COMMAND test_backend_conformance)

    # golden 在 Vulkan 上复跑（M1.4d）：同一份 PyTorch 黄金数据，未支持算子的用例自动跳过
    add_test(NAME golden_vulkan COMMAND test_golden)
    set_tests_properties(golden_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")
    add_test(NAME grad_golden_vulkan COMMAND test_grad_golden)
    set_tests_properties(grad_golden_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # autograd 全套（gradcheck + 端到端 MLP 训练）在 Vulkan 上复跑（M1.4e）
    add_test(NAME grad_vulkan COMMAND test_grad)
    set_tests_properties(grad_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 优化器（主机端更新 + Vulkan 张量）在 Vulkan 上复跑（M1.5a）
    add_test(NAME optim_vulkan COMMAND test_optim)
    set_tests_properties(optim_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 分配生命周期在 Vulkan 上复跑（M2.1d）
    add_test(NAME alloc_vulkan COMMAND test_alloc)
    set_tests_properties(alloc_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # FIX-003 / Q26：强制 host-visible 内存路径（回退/旧行为基线，覆盖 TRC_VK_HOST_MEMORY）
    add_test(NAME alloc_vulkan_host_memory COMMAND test_alloc)
    set_tests_properties(alloc_vulkan_host_memory PROPERTIES
                         ENVIRONMENT "TRC_TEST_DEVICE=vulkan;TRC_VK_HOST_MEMORY=1")

    # FIX-003 修订（2026-09-27）：device-local 为显式 opt-in，覆盖 TRC_VK_DEVICE_LOCAL=1 路径
    add_test(NAME alloc_vulkan_device_local COMMAND test_alloc)
    set_tests_properties(alloc_vulkan_device_local PROPERTIES
                         ENVIRONMENT "TRC_TEST_DEVICE=vulkan;TRC_VK_DEVICE_LOCAL=1")

    # 规模压力套件在 Vulkan 上复跑（M2.2d；默认小档）
    add_test(NAME stress_vulkan COMMAND test_stress)
    set_tests_properties(stress_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # nn 层（含 gradcheck 与端到端）在 Vulkan 上复跑（M1.5b）
    add_test(NAME nn_vulkan COMMAND test_nn)
    set_tests_properties(nn_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 损失函数（含 gradcheck 与端到端）在 Vulkan 上复跑（M1.5c）
    add_test(NAME loss_vulkan COMMAND test_loss)
    set_tests_properties(loss_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 逐 epoch 训练轨迹在 Vulkan 上复跑（M2.0d）
    add_test(NAME train_trace_vulkan COMMAND test_train_trace)
    set_tests_properties(train_trace_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # checkpoint（训练 + 序列化）在 Vulkan 张量上复跑（M1.6b）
    add_test(NAME checkpoint_vulkan COMMAND test_checkpoint)
    set_tests_properties(checkpoint_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 双图双优化器 GAN 在 Vulkan 上复跑（M2.3e）
    add_test(NAME gan_vulkan COMMAND test_gan)
    set_tests_properties(gan_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # 二期验收在 Vulkan 上复跑（M2.4）
    add_test(NAME acceptance_vulkan COMMAND test_acceptance)
    set_tests_properties(acceptance_vulkan PROPERTIES ENVIRONMENT "TRC_TEST_DEVICE=vulkan")

    # FIX-004 / Q28：分块提交（TRC_VK_CHUNK_NODES）与整图单次提交逐位一致
    add_test(NAME golden_vulkan_chunked COMMAND test_golden)
    set_tests_properties(golden_vulkan_chunked PROPERTIES
                         ENVIRONMENT "TRC_TEST_DEVICE=vulkan;TRC_VK_CHUNK_NODES=3")
    add_test(NAME grad_golden_vulkan_chunked COMMAND test_grad_golden)
    set_tests_properties(grad_golden_vulkan_chunked PROPERTIES
                         ENVIRONMENT "TRC_TEST_DEVICE=vulkan;TRC_VK_CHUNK_NODES=3")
    add_test(NAME backend_conformance_chunked COMMAND test_backend_conformance)
    set_tests_properties(backend_conformance_chunked PROPERTIES
                         ENVIRONMENT "TRC_VK_CHUNK_NODES=5")

    # 越界索引中止死亡测试（M2.1c）：由 expect_abort.cmake 判定——
    # 中止（非 0 退出）= 通过；未中止 = 失败；无设备（77）= 跳过
    add_executable(test_vulkan_index_oob test_vulkan_index_oob.cpp)
    target_link_libraries(test_vulkan_index_oob PRIVATE traincpp)
    add_test(NAME vulkan_index_oob_abort
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_vulkan_index_oob>
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

    # "支持声明"死亡测试（M2.1f / R6）：F32→I32 cast 在 Vulkan 必须中止
    add_executable(test_supports_guard test_supports_guard.cpp)
    target_link_libraries(test_supports_guard PRIVATE traincpp)
    add_test(NAME vulkan_cast_type_abort
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_supports_guard> -DARGS=cast_i32
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)

    # 设备上限死亡测试（M2.1g / R5）：descriptor range 超 maxStorageBufferRange 必须中止
    # 用 TRC_VK_MAX_STORAGE_RANGE 强制 4096 字节上限以在小设备上可复现
    add_executable(test_vulkan_limit_guard test_vulkan_limit_guard.cpp)
    target_link_libraries(test_vulkan_limit_guard PRIVATE traincpp)
    add_test(NAME vulkan_storage_range_abort
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_vulkan_limit_guard> -DARGS=storage_range
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
    set_tests_properties(vulkan_storage_range_abort PROPERTIES
                         ENVIRONMENT "TRC_VK_MAX_STORAGE_RANGE=4096")

    # 提交失败诊断死亡测试（FIX-004 / Q28）：用 TRC_VK_FORCE_SUBMIT_RESULT 伪造结果，
    # 校验中止消息含 VkResult 名称/数值；device-lost 走专用消息
    add_test(NAME vulkan_device_lost_abort
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_vulkan_limit_guard> -DARGS=submit_result
                     "-DEXPECT_REGEX=VkResult=VK_ERROR_DEVICE_LOST/-4"
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
    set_tests_properties(vulkan_device_lost_abort PROPERTIES
                         ENVIRONMENT "TRC_VK_FORCE_SUBMIT_RESULT=-4")
    add_test(NAME vulkan_submit_result_abort
             COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:test_vulkan_limit_guard> -DARGS=submit_result
                     "-DEXPECT_REGEX=VkResult=VK_ERROR_UNKNOWN/-13"
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_abort.cmake)
    set_tests_properties(vulkan_submit_result_abort PROPERTIES
                         ENVIRONMENT "TRC_VK_FORCE_SUBMIT_RESULT=-13")
endif()
