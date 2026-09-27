# train.cpp - 示例
add_executable(example_basic example_basic.cpp)
target_link_libraries(example_basic PRIVATE traincpp)

# 完整训练链路示例：nn 层 + autograd + AdamW + checkpoint + GGUF 导出
add_executable(example_train example_train.cpp)
target_link_libraries(example_train PRIVATE traincpp)

# 图像分类演示：训练一个小 CNN 并导出模型（可用 CPU 或 Vulkan）
add_executable(demo_image_train demo_image_train.cpp)
target_link_libraries(demo_image_train PRIVATE traincpp)

# 图像分类演示：加载模型并对图片分类
add_executable(demo_image_classify demo_image_classify.cpp)
target_link_libraries(demo_image_classify PRIVATE traincpp)
