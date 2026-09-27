# 第三方许可与致谢

本项目的自有代码以 MIT 许可发布，见 [LICENSE](LICENSE)。

本项目在**数据类型取值、张量内存布局、算子语义、GGUF 读写与量化反量化**等方面参考并对齐
**ggml v0.23.0**，其中部分实现（例如 fp16/bf16 位级转换、GGUF 读写、量化块布局与反量化）
参考或移植自 ggml。根据 MIT 许可的要求，以下保留 ggml 的版权与许可声明。

- 项目：ggml
- 版权：Copyright (c) 2023-2026 The ggml authors
- 许可：MIT

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
