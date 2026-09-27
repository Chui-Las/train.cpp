# API 参考

公共头位于 `include/traincpp/`。包含总头 `traincpp/traincpp.h` 即可使用全部接口。本文按头文件列出主要 API；语义细节见对应专题文档。

## 总头

| 头文件 | 内容 |
|---|---|
| `traincpp.h` | 聚合包含下面全部头文件 |

## trc_types.h

| 接口 | 说明 |
|---|---|
| `enum Type` | 数据类型枚举，取值与 ggml 一致 |
| `type_blck_size` / `type_size` / `type_sizef` / `type_row_size` | 块大小 / 单块字节 / 平均元素字节 / 行字节 |
| `type_name` | 类型名 |
| `type_is_quantized` / `type_is_supported` | 是否量化 / 是否可分配 |

## trc_tensor.h

| 接口 | 说明 |
|---|---|
| `struct Tensor` | 张量：`type`、`ne[4]`、`nb[4]`、`op`、`op_params`、`flags`、`src[10]`、`nsrc`、`view_src`、`data`、`name`、`grad`、`grad_acc` |
| `enum TensorFlags` | `TENSOR_FLAG_INPUT` / `OUTPUT` / `PARAM` / `LOSS` / `COMPUTE` |
| `new_tensor_1d/2d/3d/4d`、`new_tensor_nd` | 创建张量 |
| `tensor_dup_meta` | 复制元数据（形状/类型/名字） |
| `tensor_n_dims` / `nelements` / `nrows` / `nbytes` / `element_size` / `row_size` | 形状与大小 |
| `tensor_is_contiguous` / `are_same_shape` / `can_repeat` / `is_view` | 形状判定 |
| `tensor_offset_linear` | 线性索引到字节偏移 |
| `tensor_set_name` | 设置名字 |
| `tensor_data_f32` | F32 数据指针 |

## trc_context.h

| 接口 | 说明 |
|---|---|
| `ContextParams{mem_size, mem_buffer, no_alloc}` | 上下文参数 |
| `context_new(size_t)` / `context_new(const ContextParams&)` / `context_free` | 创建 / 释放 |
| `context_mem_size` / `mem_used` / `mem_available` | arena 统计 |
| `context_no_alloc` / `context_set_no_alloc` | 查询 / 设置分配模式 |
| `context_alloc` | 从 arena 分配 |
| `context_tensor_count` / `context_tensor` | 遍历张量 |
| `context_release_graph_tensors` | 回收图中间量 |
| `ContextMemoryStats{...}` / `context_memory_stats` | 内存核算 |

## trc_graph.h

| 接口 | 说明 |
|---|---|
| `struct Graph` | 计算图 |
| `graph_new` / `graph_free` / `graph_clear` | 创建 / 释放 / 清空 |
| `graph_build_forward_expand` | 后序展开前向图 |
| `graph_n_nodes` / `graph_node` / `graph_n_leafs` / `graph_leaf` | 遍历节点与叶子 |

## trc_backend.h

| 接口 | 说明 |
|---|---|
| `enum class DeviceType` | `CPU` / `GPU` / `IGPU` / `ACCEL` |
| `DeviceProps` | 设备信息 |
| `class Device` | `props` / `default_buffer_type` / `alloc_buffer` / `supports_op` / `graph_compute` |
| `class BufferType` / `class Buffer` | 缓冲类型与缓冲 |
| `graph_first_unsupported` | 整图预检 |
| `device_count` / `device_get` / `device_by_type` / `device_cpu` | 设备枚举 |
| `backend_register_device` | 后端注册 |
| `buffer_alloc_ctx_tensors` / `buffer_alloc_graph_tensors` / `buffer_free` | 分配 / 释放 |
| `AllocStats` / `backend_alloc_stats` / `backend_alloc_stats_reset` | 进程级分配统计 |
| `tensor_set` / `tensor_get` | 张量数据读写（字节语义） |

## trc_autograd.h

| 接口 | 说明 |
|---|---|
| `tensor_set_param` / `tensor_clear_param` | 标记 / 取消标记参数 |
| `tensor_set_loss` | 标记损失 |
| `graph_build_backward_expand` | 构建反向图 |
| `graph_get_grad` / `graph_get_grad_acc` | 取梯度 / 累加器 |
| `graph_reset` / `graph_reset_accumulate` | 清零并播种 / 仅播种 |

## trc_op.h

| 接口 | 说明 |
|---|---|
| `enum Op` | 算子枚举 |
| `op_name` / `op_symbol` / `op_desc` | 算子名 / 符号 / 描述 |

## trc_ops.h

算子构造函数。完整清单与语义见《算子参考》。

- 形状：`dup`、`view_1d..4d`、`reshape_1d..4d`、`permute`、`transpose`、`cont`、`cast`
- 逐元素二元：`add`、`add1`、`sub`、`mul`、`div`
- 逐元素一元：`neg`、`abs_op`、`sqr`、`sqrt_op`、`exp_op`、`log_op`、`sin_op`、`cos_op`、`clamp`、`sgn`、`step`
- 激活：`relu`、`leaky_relu`、`sigmoid`、`tanh_op`、`silu`、`gelu`、`gelu_erf`、`erf_op`、`softplus`、`hardswish`
- 缩放：`scale`、`scale_bias`
- 归约：`sum`、`sum_rows`、`mean`
- 归一化：`norm`、`rms_norm`、`group_norm`、`norm_back`、`rms_norm_back`、`group_norm_back`
- 注意力：`soft_max`、`soft_max_ext`
- 损失：`cross_entropy_loss`、`cross_entropy_loss_back`
- 索引：`get_rows`、`get_rows_back`、`set_rows`
- 形状扩展：`repeat`、`repeat_back`、`concat`、`pad`、`pad_ext`、`pad_back`、`acc`
- 卷积与池化：`im2col`、`im2col_back`、`conv_1d`、`conv_2d`、`col2im_1d`、`conv_transpose_1d`、`conv_transpose_2d`、`pool_2d`、`pool_2d_back`
- 线代：`mul_mat`
- 枚举：`PadMode`（`PAD_ZERO` / `PAD_REFLECT`）、`PoolMode`（`POOL_MAX` / `POOL_AVG`）

## trc_rng.h

| 接口 | 说明 |
|---|---|
| `struct Rng{state}` | 随机数状态 |
| `rng_seed` | 设置种子 |
| `rng_next_u64` / `rng_next_u32` | 下一个随机数 |
| `rng_uniform01` / `rng_uniform` / `rng_normal` | 分布采样 |
| `rng_fill_constant` / `rng_fill_uniform` / `rng_fill_normal` | 填充张量（支持延迟填充） |

## trc_nn.h

| 接口 | 说明 |
|---|---|
| `ParamList` 与 `param_list_add` / `count` / `get` / `set_param` / `merge` | 参数列表 |
| `nn_fan_in` / `nn_fan_out` | fan 计算 |
| `nn_init_constant` / `uniform` / `normal` / `kaiming_uniform` | 初始化 |
| `Linear` | `linear_init` / `linear_init_weight` / `linear_forward` / `linear_forward_weight` / `linear_params` |
| `Embedding` | `embedding_init` / `embedding_init_weight` / `embedding_forward` / `embedding_params` |
| `Conv1d` | `conv1d_init` / `conv1d_init_groups` / `conv1d_init_weight` / `conv1d_forward` / `conv1d_forward_weight` / `conv1d_params` |
| `Conv2d` | `conv2d_init` / `conv2d_init_weight` / `conv2d_forward` / `conv2d_forward_weight` / `conv2d_params` |
| `ConvTranspose1d` | `convtranspose1d_init` / `convtranspose1d_init_weight` / `convtranspose1d_forward` / `convtranspose1d_forward_weight` / `convtranspose1d_params` |
| `WeightNorm` | `weightnorm_init` / `weightnorm_forward` / `weightnorm_params` / `weightnorm_sync_g` |
| `LayerNorm` | `layernorm_init` / `layernorm_init_weight` / `layernorm_forward` / `layernorm_params` |
| `RmsNorm` | `rmsnorm_init` / `rmsnorm_init_weight` / `rmsnorm_forward` / `rmsnorm_params` |
| `GroupNorm` | `groupnorm_init` / `groupnorm_init_weight` / `groupnorm_forward` / `groupnorm_params` |
| `Dropout` | `dropout_init` / `dropout_set_training` / `dropout_forward` / `dropout_refresh` / `dropout_params` |

## trc_loss.h

| 接口 | 说明 |
|---|---|
| `mse_loss` | 全元素均值 MSE |
| `l1_loss` | 全元素均值 L1 |

## trc_optim.h

| 接口 | 说明 |
|---|---|
| `SgdOptions` / `AdamwOptions` | 优化器选项 |
| `optim_sgd_new` / `optim_adamw_new` / `optim_free` | 创建 / 释放 |
| `optim_add_param_group_sgd` / `optim_add_param_group_adamw` | 参数分组 |
| `optim_group_count` / `optim_param_count` / `optim_param` | 分组 / 参数查询 |
| `optim_state_count` / `optim_state` / `optim_alloc_state` | 状态查询 / 分配 |
| `optim_group_type` / `optim_group_param_count` / `optim_group_state_count` / `optim_get_lr_group` / `optim_group_sgd_options` / `optim_group_adamw_options` | 分组信息 |
| `optim_step_count` / `optim_set_step_count` / `optim_state_step` / `optim_set_state_step` | 步数 |
| `optim_get_lr` / `optim_set_lr` / `optim_set_lr_group` | 学习率 |
| `optim_zero_grad` / `optim_step` / `optim_clip_grad_norm` | 更新 |
| `lr_scheduler_step_new` / `exponential_new` / `cosine_new` / `warmup_cosine_new` / `free` | 调度器创建 / 释放 |
| `lr_scheduler_step` / `get_lr` / `epoch` | 调度器推进 / 查询 |

## trc_gguf.h

| 接口 | 说明 |
|---|---|
| `enum class GgufType` | 元数据类型 |
| `GgufTensorInfo` | 张量元信息 |
| `gguf_open` / `gguf_close` | 打开 / 关闭 |
| `gguf_version` / `alignment` / `data_offset` | 头信息 |
| `gguf_kv_count` / `find_key` / `key` / `kv_type` / `arr_type` / `arr_count` / `has_key` | 元数据查询 |
| `gguf_get_u8/i8/u16/i16/u32/i32/f32/bool/u64/i64/f64/str` | 标量取值 |
| `gguf_get_arr_i32/i64/f32/str` | 数组取值 |
| `gguf_tensor_count` / `find_tensor` / `tensor_info` / `tensor_nelements` | 张量查询 |
| `gguf_tensor_to_f32` / `gguf_load_tensor` / `gguf_new_tensor` | 读取 / 反量化 / 建张量 |
| `gguf_validate` | 结构自检 |
| `gguf_writer_new` / `free` | 写出器 |
| `gguf_writer_set_arch` / `set_u8..f64` / `set_str` / `set_arr_i32/i64/f32/str` | 元数据设置 |
| `gguf_writer_add_tensor` / `tensor_count` | 添加张量 |
| `gguf_writer_write` | 写出文件 |

## trc_checkpoint.h

| 接口 | 说明 |
|---|---|
| `CheckpointOptimizer{name, optimizer, scheduler}` | 单优化器分区 |
| `CheckpointItems{optimizer, rng, scheduler, optimizers, optimizer_count}` | 保存 / 载入项 |
| `CheckpointInfo{...}` | 探测结果 |
| `checkpoint_save` / `checkpoint_load` | 保存 / 载入 |
| `checkpoint_probe` / `checkpoint_verify` | 探测 / 自检 |
