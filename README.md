# FPGA竞赛_2026 · U126 KV260 Attention 加速器

发布日期：2026-11-04。Built with Llama。

独立竞赛仓库：[yanghaoju/FPGA_2026](https://github.com/yanghaoju/FPGA_2026)。
本次代码 RAR 和单独的模型权重 RAR 统一在 [U126 提交版发布页](https://github.com/yanghaoju/FPGA_2026/releases/tag/u126-submission-20261104)提供。

本包对应正式 U126 L512，包含已通过 2026-11-02 板端回归的输入打包与输出解包优化。硬件位流保持正式 U126；跨模型、LUT 和其他架构候选不在本版内。包内保留源码、板端运行文件、测试、构建与运行入口、许可证及身份校验文件。

## 内容与环境

- `hardware/U126/`：联合 Prefill/Decode IP 的 HLS C++、头文件、测试平台和 Tcl。
- `llama.cpp/`：可构建的 llama.cpp 源码、依赖、U126 hook 与 XRT session。不是只有补丁。
- `board/`：配套 XCLBIN、设备树、AArch64 llama-server 和动态库。
- `validation/`：数值测试源码、已编译测试程序和冻结输入/参考输出。
- `scripts/`：校验、构建、安装、启动、真实请求演示及模型下载入口。
- `model/`：权重来源、固定 revision、大小和 SHA-256；完整权重另包提供。

目标环境：KV260/K26，AArch64 Ubuntu 22.04.5，XRT 2.13.0；安装与加载使用 `xmutil`。硬件重建使用 Vitis/Vivado/Vitis HLS 2022.2 及与板端镜像匹配的 KV260 `.xpfm` 平台。该平台和操作系统镜像须由开发环境提供，不包含在代码包中。

正式配置是 32 Q heads / 8 KV heads / head_dim 64，最大 KV 长度 512；Prefill 在容量内按实际有效长度执行。512 是容量上限，并非只能输入固定几个长度。运行时使用 BF16 张量 I/O、FP32 Attention 计算及在线 softmax 状态；历史 kernel 名中的 `int8` 不表示本版使用 INT8 Attention 算术。

## 模型权重

完整模型演示需要 `Llama-3.2-1B-Instruct-Q4_K_M.gguf`，807,694,464 字节，不能放入 200 MB 的代码提交包。它属于 PS 端完整模型推理所用权重，PL Attention IP 接收的是 Q/K/V 张量。只运行 `validation` 中的 IP 数值测试无需 GGUF。

优先使用单独提供的 `U126_Model_Llama32_1B_Q4_K_M_20261104.rar`，解压后取其中 `model/` 下的 GGUF；或运行：

```bash
python3 scripts/download_u126_model.py
# 如需改用镜像：python3 scripts/download_u126_model.py --mirror
python3 scripts/verify_u126_release.py --model /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

模型继续受 Llama 3.2 Community License 与 Acceptable Use Policy 约束，见 `licenses/` 和 `Notice`。权重校验值：

```text
6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83
```

## 直接部署与运行

把代码目录和单独的模型文件拷到 KV260，进入代码目录。加载前停止占用 PL/XRT 的程序；若另一应用已加载，先用 `xmutil listapps` 查看并手动卸载对应 slot。

```bash
python3 scripts/verify_u126_release.py
sudo bash scripts/install_u126_payload.sh
sudo bash scripts/activate_u126_overlay.sh --reload
bash scripts/run_u126_numeric_gate.sh
bash scripts/start_u126_demo.sh /path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf 8083
```

另开板端终端：

```bash
bash scripts/smoke_u126_server.sh 8083
python3 scripts/demo_u126_request.py --port 8083
```

演示入口会显示真实模型回答、输入/输出 token 数、耗时及 Prefill/Decode tokens/s；这些现场数字是完整模型服务口径。Attention 数值门禁会报告其自身 kernel 与 API 耗时。不要混用两种口径。服务默认仅监听 `127.0.0.1`，可在板端运行演示脚本；需要访问时可自行配置受控连接。

启动脚本固定 strict-PL、BF16 KV cache、单请求、总上下文 512、关闭 context shift。提示词与生成预算合计须留在 512 容量内，演示脚本使用更保守的 504 上限。正常停止服务后，查看 `/tmp/u126_pl_report_8083.txt`，应有 PL 调用，且 `fallback=0`、`failure=0`。

## 从源码构建

KV260 上安装 CMake 3.18 或更新版本、`g++`、OpenMP 和 XRT 开发头文件/库后（Ubuntu 22.04 自带的 CMake 版本满足要求）：

```bash
bash scripts/build_u126_runtime.sh
# 数值测试程序也可单独从源码编译：先停止服务
bash scripts/run_u126_numeric_gate.sh --build
```

新运行时输出到 `build/llama-u126/bin/`。该构建不会覆盖已验证的 `/opt/u126` 运行文件。若要测试重建的服务程序，使用：

```bash
U126_PORT=8084 bash src/run_u126_server_template.sh \
  build/llama-u126/bin/llama-server /path/to/model.gguf \
  /lib/firmware/xilinx/fpt-u126-dynamic-34fd414e/binary_container_1.bin \
  build/llama-u126/bin
```

开发机上加载 2022.2 工具环境并指定匹配的平台后，可重建硬件：

```bash
export FPT_KV260_XPFM=/absolute/path/to/k26_base_starter_kit.xpfm
bash scripts/build_u126_hardware.sh
```

脚本依次运行 HLS 测试、综合/导出和 Vitis link，输出到 `build/hardware-u126/`。重建结果应重新检查时序、数值与板端兼容性，再用于部署。

## 版本与校验

`SHA256SUMS.txt` 是代码包的文件校验清单；`board/runtime_manifest.json` 固定板端二进制身份，`SOURCE_PROVENANCE.json` 记录源码快照和集成来源。现成运行库的确切上游 llama.cpp 提交号无法恢复，因此不声称源码重编译与现成二进制逐字节一致。

随包预编译文件沿用已完成的板端回归：L128 Prefill + 8 步 Decode、L512 Prefill、logits/replay-token 身份检查及 strict-PL 真实服务 smoke。摘要见 `validation/release_validation.json`。本次重新打包另做文件身份、脚本语法、源码 CPU 构建、同一 GGUF 的 CPU 冒烟推理和压缩包完整性校验。CPU 检查用于验证源码依赖与模型文件，不代替 XRT/PL 回归，也不作为板端性能数据。
