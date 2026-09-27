// train.cpp - CPU 参考后端
// 设计原则：正确性优先（标量循环），性能优化留到后续里程碑。
#include "trc_cpu.h"

#include "core/trc_impl.h"

namespace traincpp {

namespace {

constexpr size_t CPU_BUFFER_ALIGN = 64;

// ---------------------------------------------------------------- buffer
class CpuBuffer;

class CpuBufferType : public BufferType {
public:
    Device* device() const override;
    const char* name() const override { return "CPU"; }
    size_t alignment() const override { return CPU_BUFFER_ALIGN; }
    Buffer* alloc_buffer(size_t size) override;
};

class CpuBuffer : public Buffer {
public:
    CpuBuffer(CpuBufferType* buft, size_t size) : buft_(buft), size_(size) {
        data_ = alloc_aligned(size_ == 0 ? CPU_BUFFER_ALIGN : size_, CPU_BUFFER_ALIGN);
        TRC_ASSERT(data_ != nullptr,
                   "CpuBuffer: 内存申请失败：需要 %zu 字节（%.2f MiB）；请减小 batch/segment "
                   "或检查可用物理内存",
                   size_, (double) size_ / (1024.0 * 1024.0));
        std::memset(data_, 0, size_);
    }

    ~CpuBuffer() override { free_aligned(data_); }

    BufferType* buffer_type() const override { return buft_; }
    size_t size() const override { return size_; }
    void* base() override { return data_; }
    bool is_host() const override { return true; }

    void set_tensor(Tensor* t, size_t offset, const void* data, size_t size) override {
        TRC_ASSERT(t->data != nullptr, "CpuBuffer::set_tensor: 张量未分配数据");
        std::memcpy((uint8_t*) t->data + offset, data, size);
    }

    void get_tensor(const Tensor* t, size_t offset, void* data, size_t size) const override {
        TRC_ASSERT(t->data != nullptr, "CpuBuffer::get_tensor: 张量未分配数据");
        std::memcpy(data, (const uint8_t*) t->data + offset, size);
    }

    void clear() override { std::memset(data_, 0, size_); }

private:
    CpuBufferType* buft_;
    size_t         size_;
    uint8_t*       data_ = nullptr;
};

Buffer* CpuBufferType::alloc_buffer(size_t size) {
    return new CpuBuffer(this, size);
}

// ---------------------------------------------------------------- device
class CpuDevice : public Device {
public:
    CpuDevice() {
        props_.name = "CPU";
        props_.description = "CPU 参考实现（标量循环，正确性优先）";
        props_.type = DeviceType::CPU;
        props_.unified_memory = true;
        props_.memory_free = 0;
        props_.memory_total = 0;
    }

    const DeviceProps& props() const override { return props_; }
    BufferType* default_buffer_type() override { return &buft_; }

    bool supports_op(const Tensor* t) const override { return cpu_supports_op(t); }

    bool graph_compute(Graph* graph) override {
        TRC_ASSERT(graph != nullptr, "CpuDevice::graph_compute: graph 为空");
        for (Tensor* node : graph->nodes) {
            cpu_compute_node(node);
        }
        return true;
    }

private:
    DeviceProps  props_;
    CpuBufferType buft_;
};

} // namespace

Device* cpu_device() {
    static CpuDevice device;
    return &device;
}

Device* CpuBufferType::device() const {
    return cpu_device();
}

} // namespace traincpp
