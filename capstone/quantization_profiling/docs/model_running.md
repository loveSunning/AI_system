# 本地模型运行与 vLLM / Transformers 学习指南

目标环境：Linux、RTX 4090D、约20GiB可用显存。当前统一环境采用 vLLM 0.14.1、torch 2.9.1+cu128、Transformers 4.57.3 和 LLM Compressor 0.9.0.5；目录仍沿用 vllm-0112。用户提供的历史BF16结果来自原 vLLM 0.11.0/torch 2.8.0+cu129/Transformers 4.57.6，保留独立记录。

安装、镜像和量化说明见 [统一环境入口](../README.md#vllm-0112-统一量化与推理环境)。旧0.11.0的本地分块FP8加载曾出现 orig_dtype 异常；当前代码的版本预检已接受完整统一组合，实际GPU转换、加载和性能仍需验证。

入口为 [run_model.py](../scripts/run_model.py)。它支持本地文本因果语言模型：Transformers 入口运行未量化模型，vLLM 入口运行受该版本支持的普通/AWQ/FP8 模型。模型目录与量化配置必须对应；更换 `--dtype` 不会把 BF16 权重转换成 AWQ。视觉输入、工具调用和正式性能评测不属于这个入口。

## 1. 先跑通一个短请求

从仓库根目录执行。以下各条推理命令分别运行，前一个模型进程退出后再启动另一个；两个进程同时加载 8B 模型会争用显存。

### 1.1 只看模板和 token，暂不加载权重

```bash
python capstone/quantization_profiling/scripts/run_model.py \
  --model /workspace/weights_pub/Qwen3-8B \
  --thinking off \
  --prompt "请用三句话介绍 KV cache。" \
  --tokenize-only \
  --output out/quantization_profiling/tokenize.json
```

观察渲染后的 Prompt、`input_tokens` 和 JSON 中的 `input_token_ids`。中文字符数不等于 token 数；system 消息、角色分隔符和生成提示也占输入 token。此模式需要 Transformers/tokenizer 依赖，但不需要 vLLM、CUDA 或模型权重加载。

### 1.2 用 Transformers 理解完整生成链路

```bash
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend transformers \
  --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 --thinking off \
  --max-model-len 2048 --max-new-tokens 128 \
  --prompt "请用三句话解释 prefill 和 decode。" \
  --output out/quantization_profiling/transformers-smoke.json
```

脚本使用当前环境的 PyTorch 和 Accelerate 将模型放到一张 CUDA 设备上；不会自动转到 CPU。需要 CPU 学习时，可对更小的本地模型显式使用 `--device cpu --dtype float32`，并准备足够内存。

### 1.3 用 vLLM 跑同一个请求

```bash
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm \
  --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 --thinking off \
  --max-model-len 2048 --max-new-tokens 128 \
  --gpu-memory-utilization 0.75 --max-num-seqs 1 \
  --prompt "请用三句话解释 prefill 和 decode。" \
  --output out/quantization_profiling/vllm-smoke.json
```

vLLM 入口默认启用 eager 执行、关闭 prefix caching，并把每步调度 token 预算限制为 2048、启用 chunked prefill，避免第一次学习时混入 CUDA Graph 捕获和前缀缓存命中。后续需要研究 Graph 时再显式使用 `--no-enforce-eager`。0.75 是物理总显存的比例：24 GiB 显卡约对应 18 GiB 的引擎预算，不是“20 GiB 空闲显存的 75%”；它也不是整卡硬隔离上限。权重、KV 和运行时开销都需容纳在实际环境中，短请求参数不能保证一定不 OOM。

### 1.4 四种模型使用同一个入口

`--model` 既接受本地路径，也接受别名；别名从 `MODEL_ROOT`（默认 `/workspace/weights_pub`）查找对应目录。

| 别名 | 子目录 | 预设执行方案 |
| --- | --- | --- |
| qwen3-8b | Qwen3-8B | BF16，无权重量化 |
| qwen3-8b-awq | Qwen3-8B-AWQ | awq_marlin，BF16 激活、INT4 权重 |
| qwen3-8b-fp8 | Qwen3-8B-FP8 | 官方分块 FP8，原生 fp8 加载器 |
| qwen3-8b-fp8-local | Qwen3-8B-FP8-local | BF16 非量化部分，按 config 识别 FP8 量化层 |

```bash
# 已下载的 AWQ 权重：使用支持 BF16 的 Marlin 路径。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --model qwen3-8b-awq --thinking off \
  --max-model-len 2048 --max-new-tokens 128 \
  --prompt "请用三句话解释权重量化。" \
  --output out/quantization_profiling/awq-smoke.json

# 官方分块 FP8：使用原生 fp8 权重配置。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --model qwen3-8b-fp8 --thinking off \
  --prompt "请简要解释动态激活量化。" \
  --output out/quantization_profiling/fp8-official-smoke.json

# 自行转换的本地 FP8 权重：直接读取 config 的 compressed-tensors 方案。
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --model qwen3-8b-fp8-local --thinking off \
  --prompt "请简要解释动态激活量化。" \
  --output out/quantization_profiling/fp8-smoke.json

# 四者均可启动服务；同一端口每次只运行一种模型。
python capstone/quantization_profiling/scripts/run_model.py \
  --serve --model qwen3-8b-fp8-local \
  --host 0.0.0.0 --port 7081 --max-num-seqs 1
```

脚本根据模型配置解析预设，并检查声明的模型与权重量化信息是否一致。AWQ 预设使用 `awq_marlin`，以便与 BF16 基线保持相同激活 dtype。若确实要测试旧 AWQ 路径，显式使用 `--quantization awq --dtype float16`；该路径和默认 Marlin 路径应分组比较。官方 FP8 使用原生 `fp8` 加载器；对于本地 FP8，保留 `quant_method=compressed-tensors`，不要手动改成 `fp8`；`--quantization auto` 让 vLLM 根据实际权重配置加载。

启动日志可核验实际引擎 dtype 和量化方法，具体执行 kernel 仍需 Profiler 证据。Transformers 入口会拒绝已量化模型；本次四模型对比使用 vLLM。正式比较使用 [benchmark 流程](week1_execution.md)，共用冻结请求，不能直接把上述单次短请求耗时当成性能结论。

## 2. 通用脚本的输入、参数和输出

### 输入方式

`--prompt` 可重复，表示多个独立请求，不表示同一段对话的连续轮次：

```bash
python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --thinking off \
  --system "你是一位用简洁中文回答的老师。" \
  --prompt "什么是 tokenizer？" \
  --prompt "什么是 KV cache？"
```

一个长提示词可写入 UTF-8 文件并用 `--prompt-file /path/to/prompt.txt` 读取，整份文件作为一个请求。多轮对话用 `--messages-file /path/to/messages.json`，文件内容例如：

```json
[
  {"role": "system", "content": "你是一位用简洁中文回答的老师。"},
  {"role": "user", "content": "什么是 KV cache？"},
  {"role": "assistant", "content": "它缓存已处理 token 的 Key 和 Value，供后续生成复用。"},
  {"role": "user", "content": "为什么它会随着上下文增长？"}
]
```

三种输入方式互斥；消息文件只接受 system/user/assistant 的文本消息，末尾为 user。没有提供输入时使用内置短问题。基础模型没有 chat template 时使用 `--raw-prompt`，直接续写文本。

### 参数速查

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| --backend | vllm | 选择推理库 |
| --model | /workspace/weights_pub/Qwen3-8B | 本地目录或 qwen3-8b / qwen3-8b-awq / qwen3-8b-fp8 / qwen3-8b-fp8-local；MODEL_ROOT 改变别名根目录 |
| --tokenizer | 与 model 相同 | tokenizer 的本地目录 |
| --dtype | auto | 按模型预设解析；上述四模型主线使用 BF16；不是量化算法 |
| --thinking | auto | 保留模板默认值；on/off 显式传入 enable_thinking |
| --max-new-tokens | 128 | 生成 token 上限，不保证一定生成到该长度 |
| --max-model-len | 2048 | 输入加输出的总 token 预算；vLLM 同时用它限制引擎上下文 |
| --temperature | 0.7 | 采样温度；0 切换为 greedy |
| --top-p / --top-k | 0.8 / 20 | 核采样阈值、候选 token 数；top-k=0 表示关闭 top-k |
| --repetition-penalty | 1.0 | 重复惩罚，1.0 不额外惩罚 |
| --seed | 42 | 随机种子；不同引擎/硬件不保证逐 token 一致 |
| --gpu-memory-utilization | 0.75 | vLLM 每张卡按总显存计算的预算比例 |
| --max-num-seqs / --concurrency | 1 | vLLM 同时调度的序列数上限；不限制输入列表长度，也不是 HTTP 连接数 |
| --max-num-batched-tokens | 2048 | vLLM 每步调度 token 预算，含 prefill；配合 chunked prefill 控制单步工作量 |
| --tensor-parallel-size | 1 | vLLM tensor parallel 卡数 |
| --enforce-eager / --no-enforce-eager | eager | 控制 vLLM 是否禁用 CUDA Graph 捕获 |
| --quantization | auto | 按模型预设/权重配置选择；AWQ 主线 awq_marlin，官方 FP8 使用 fp8，本地 FP8 自动识别 compressed-tensors |
| --device | cuda | Transformers 使用 cuda 或 cpu |
| --tokenize-only | 关闭 | 仅准备模板/token，不加载权重 |
| --output | 不写文件 | 指定 JSON 记录路径；同路径会覆盖 |
| --debug | 关闭 | 错误时输出完整 traceback |

`thinking` 是模板能力，不是所有模型都有的通用开关。Qwen3 的模板实现了 `enable_thinking`；脚本在显式 on/off 时检查模板是否包含该变量，不会给其他模型静默套用这个设置。Qwen3 示例均显式关闭 thinking，避免短输出预算主要消耗在思考内容上；开启时增加输出预算并按官方建议设置采样参数。[Qwen3 模型卡](https://huggingface.co/Qwen/Qwen3-8B)

JSON 包含模型和 tokenizer 路径、运行参数、包版本、config/template 的 SHA256、每个请求的输入与输出 token IDs、文本、结束原因和粗略耗时。config/template hash 不等于整个权重仓库的不可变 revision；仍需另外锁定模型来源。生成文本保留模型实际输出，不做思考内容解析。

`load_seconds`、`generation_seconds` 用于确认调用链路，包含首次运行开销，不是 TTFT、TPOT 或正式吞吐指标。两个框架可能以不同方式报告 EOS token；不要据此直接完成 P1 性能/质量比较。

## 3. Transformers：从文本到新 token

库名是 **Transformers**，Python 导入名是 `transformers`。它提供模型实现、分词器和通用生成接口，适合先理解一次推理的每一步。

| API | 输入和输出 | 应理解的内容 |
| --- | --- | --- |
| AutoTokenizer.from_pretrained | 本地目录 → tokenizer | 读取词表、特殊 token 与 chat template，不加载模型权重 |
| tokenizer.apply_chat_template | messages → 模型期望的对话文本/token | 添加角色分隔符与 assistant 生成起点；模板属于模型协议 |
| tokenizer(...) / encode | 文本 → input_ids、attention_mask | token ID 是模型输入；mask 区分真实输入与 padding |
| AutoModelForCausalLM.from_pretrained | 目录 → 因果语言模型 | 读取 config 和权重；Auto 根据 config 选择具体模型类 |
| model.eval / torch.inference_mode | 配置执行状态 | eval 切换推理行为；inference_mode 关闭梯度记录，两者职责不同 |
| model.generate | 输入 token → 输入与生成 token 的拼接 | 限定采样与停止条件，保留 KV cache 加速后续 token |
| tokenizer.decode | 新 token IDs → 文本 | 只解码新增部分，避免把输入当成模型回答 |

下面是脱离项目封装后的最小完整示例，单独保存为 Python 文件执行；它演示未量化模型的单条请求：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

model_dir = "/workspace/weights_pub/Qwen3-8B"
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map={"": "cuda"},
    local_files_only=True,
)
model.eval()

messages = [{"role": "user", "content": "请简要解释 KV cache。"}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to(model.device)
params = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.0,
    eos_token_id=model.generation_config.eos_token_id,
    bos_token_id=model.generation_config.bos_token_id,
    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
)
with torch.inference_mode():
    sequences = model.generate(**inputs, generation_config=params)

new_ids = sequences[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

### 容易混淆的参数

- `max_new_tokens` 只约束新增 token；`max_length` 通常包含输入。脚本统一采用前者，并单独检查“输入 + 新输出上限”。
- `temperature>0` 的采样需要 `do_sample=True`。greedy 时脚本使用 `do_sample=False`，不向 Transformers 传 temperature/top-p/top-k。
- 模型目录里的 `generation_config.json` 可能带有默认采样设置。脚本使用新的 `GenerationConfig` 明确设置本次协议，并保留模型的 BOS/EOS 停止 token。
- `device_map="auto"` 依赖 Accelerate 的设备分配，可能产生 CPU offload；这会改变速度与显存行为。当前脚本采用单设备显式映射。
- decoder-only 模型批量生成时，脚本使用左 padding 和 attention mask。切回答时按整个 batch 的填充宽度切片，不按某一条未填充的 token 数切片。
- `use_cache=True` 复用当前生成过程的 KV，不等于跨请求 prefix caching；权重 dtype、KV dtype、量化位宽也不是同一个参数。

参考 [chat template 文档](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating)、[生成接口与 GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)。上述链接使用 4.57 系列文档；统一环境固定为 4.57.3，旧环境固定为 4.57.6。

## 4. vLLM：让推理引擎管理执行

vLLM 提供模型执行、请求调度和 KV 管理。常用入口分为进程内的 `LLM` 和常驻 HTTP 服务；它们都使用底层推理引擎，适合不同调用方式。

| API / 对象 | 用法 | 作用 |
| --- | --- | --- |
| LLM(...) | 初始化一次，重复使用 | 加载权重、建立执行器与缓存；不要每问一题重建一次 |
| SamplingParams(...) | 为请求指定采样协议 | max_tokens、temperature、top-p、top-k、seed 等 |
| llm.generate(...) | 文本列表或 token prompt 列表 | 完成批量推理并返回 RequestOutput 列表 |
| llm.chat(...) | 消息列表/消息列表的列表 | 引擎应用 chat template；可传 chat_template_kwargs |
| RequestOutput.outputs[0] | 取第一个 completion | 读取 text、token_ids、finish_reason |
| API server | 模型常驻，客户端发 HTTP | 多请求持续到达时由服务端负责调度 |

最小的 token 输入示例：

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def main():
    model_dir = "/workspace/weights_pub/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    messages = [{"role": "user", "content": "请简要解释 KV cache。"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    llm = LLM(
        model=model_dir, tokenizer=model_dir, dtype="bfloat16",
        max_model_len=2048, gpu_memory_utilization=0.75, max_num_seqs=1,
        enforce_eager=True, enable_prefix_caching=False,
        max_num_batched_tokens=2048, enable_chunked_prefill=True,
        generation_config="vllm", seed=42,
    )
    params = SamplingParams(
        max_tokens=128, temperature=0.7, top_p=0.8, top_k=20,
        repetition_penalty=1.0, seed=42,
    )
    outputs = llm.generate([{"prompt_token_ids": token_ids}], sampling_params=params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` 让 worker 使用 spawn 启动时可以安全导入入口。项目脚本会在启动 vLLM 前设置默认 spawn 模式，且不提前初始化 PyTorch CUDA。

如果希望由 vLLM 处理模板，可在同一个引擎上改用：

```python
outputs = llm.chat(
    [messages],
    sampling_params=params,
    chat_template_kwargs={"enable_thinking": False},
)
print(outputs[0].outputs[0].text)
```

项目脚本采用显式 token 输入，让两个后端共用同一份模板结果，便于对照；使用 `LLM.chat` 时不要再把渲染后的文本包装成 user 消息，否则会重复套模板。vLLM 的 `generation_config="vllm"` 用于避免读取仓库生成默认值改变本次设置；greedy/关闭 top-k 时脚本把候选数转换为 vLLM 使用的 `top_k=-1`。

参考 [vLLM 0.11.0 LLM API](https://docs.vllm.ai/en/v0.11.0/api/vllm/entrypoints/llm.html)、[SamplingParams 源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py)、[Qwen 的 vLLM 示例](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)。量化方式还需匹配该版本与实际硬件支持。

## 5. 启动 HTTP 服务并用 Python 调用

### 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 python capstone/quantization_profiling/scripts/run_model.py \
  --backend vllm --serve \
  --model /workspace/weights_pub/Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 2048 --gpu-memory-utilization 0.75 --concurrency 4 \
  --host 0.0.0.0 --port 7081 --served-model-name local-model
```

脚本用当前 Python 启动 `vllm.entrypoints.openai.api_server`，打印完整命令并返回其退出码；Ctrl+C 可停止服务。服务模式不接受 `--prompt`、`--thinking`、`--max-new-tokens` 等请求参数，这些由客户端逐请求发送。`generation-config=vllm` 避免继承仓库默认采样参数；客户端应显式设置本次采样协议。

等待服务就绪，在同一机器的另一个终端检查：

```bash
curl http://127.0.0.1:7081/health
curl http://127.0.0.1:7081/v1/models
```

### 远程访问地址与接口

服务器监听 `0.0.0.0:7081`，表示绑定所有 IPv4 网卡。客户端使用 `http://服务器实际IP:7081`；`0.0.0.0` 不是远程目标地址，客户端上的 `127.0.0.1` 只会连接客户端自己。在服务器本机测试时才使用 `http://127.0.0.1:7081`。

以 `http://SERVER_IP:7081` 为根地址，替换 SERVER_IP 后访问：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | /health | 引擎健康检查；正常为 HTTP 200，响应体可为空 |
| GET | /v1/models | 查询客户端可用的模型 ID/别名 |
| POST | /v1/chat/completions | 传 messages 进行对话生成；stream=true 为 SSE |
| POST | /v1/completions | 传 prompt 做文本续写；不等同于自动套用聊天模板 |
| GET | /version | 查询服务的 vLLM 版本 |
| GET | /docs、/openapi.json | 默认启用时查看交互文档与接口 schema |
| GET | /metrics | Prometheus 指标；本测试脚本不做性能采集 |

路径依据 [vLLM 0.11.0 服务端源码](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/api_server.py)。如果服务器位于容器中，还需要将容器 7081 端口映射到宿主机，例如启动容器时使用 `-p 7081:7081`；同时确认该端口可从客户端网络到达。

### 用远程测试脚本检查

脚本为 [scripts/test_server.py](../scripts/test_server.py)。它在客户端运行，只依赖 Python 标准库，无需安装 torch、vLLM 或下载模型。以下命令中的 SERVER_IP 替换为实际服务器 IP 或域名；命令从该客户端的仓库根目录执行。

```bash
# 只检查健康状态和模型列表，不生成回答。
python capstone/quantization_profiling/scripts/test_server.py --base-url http://SERVER_IP:7081 --check-only

# 检查健康状态、模型列表和一次非流式对话生成。
python capstone/quantization_profiling/scripts/test_server.py --base-url http://SERVER_IP:7081 --model local-model

# 在普通生成之外，再发送一次流式请求，并保存结果。
python capstone/quantization_profiling/scripts/test_server.py --base-url http://SERVER_IP:7081 --model local-model --stream --output out/quantization_profiling/remote-api-check.json
```

没有指定 `--model` 时自动使用 `/v1/models` 中的第一个 ID；显式指定时会先核对该别名是否存在。`--base-url` 可带末尾的 `/v1`，脚本会规范化，避免生成重复的 `/v1/v1` 路径。

默认对 Qwen3 发送 `enable_thinking=false`、最多 128 个新 token。其他模型可使用 `--thinking auto` 省略该模板扩展；`--prompt`、`--max-tokens`、`--temperature`、`--top-p`、`--top-k` 和 `--seed` 可控制请求。服务未就绪或慢请求可设置 `--timeout 300`；该值是 socket 连接/读取超时，不是整个生成过程的总时限。

如果服务配置了 API Key，脚本从客户端的 `VLLM_API_KEY` 环境变量读取，或用 `--api-key-env` 指定变量名。结果文件不写请求认证头。如果内网访问被环境代理接管，可加 `--no-proxy` 直接连接。

成功返回退出码 0，网络/HTTP/响应协议错误返回 1，中断返回 130。无论是否指定 `--output`，都自动将 `.log`、逐请求 `.jsonl` 和完整 `.json` 报告写入客户端项目的 `log/`，失败时也保留已完成的检查。`--output` 是额外导出，`--log-dir` 可覆盖日志目录。

### 并发测试与推理统计

服务端通过 `run_model.py --serve --concurrency 4`（等同于 `--max-num-seqs 4`）设置 vLLM 调度序列上限；该参数不增加 GPU 数，也不限制 HTTP 连接数量。Transformers 离线入口仍采用输入列表的一个静态 batch，`--concurrency` 仅控制 vLLM 调度。

```bash
# 非流式：最多同时发出 4 个请求，总共完成 20 次尝试，保存所有成功和失败。
python capstone/quantization_profiling/scripts/test_server.py \
  --base-url http://SERVER_IP:7081 --model local-model \
  --concurrency 4 --requests 20 --require-server-metrics

# 流式：同样的并发与请求数，逐请求接收 usage 和服务端 profiling。
python capstone/quantization_profiling/scripts/test_server.py \
  --base-url http://SERVER_IP:7081 --model local-model \
  --stream-only --concurrency 4 --requests 20 --require-server-metrics
```

客户端每个工作线程使用独立 HTTP client；一个请求完成后再领取下一个，属于固定在途数量的闭环负载。`--requests` 未指定时等于并发数。原有 `--stream` 保持“先非流式、再流式”的行为，两个分组各发送指定次数，分别汇总。若客户端并发高于服务端调度上限，排队时间会进入耗时。

`run_model.py` 使用 [vLLM 0.11.0 支持的 ASGI middleware 扩展](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/cli_args.py) 自动加载 `profiling_middleware.py`。更新代码后需要重启服务。非流式响应新增如下字段；原有回答和 usage 保留：

```json
{
  "profiling": {
    "inference_seconds": 2.0,
    "input_tokens": 30,
    "output_tokens": 100,
    "output_tokens_per_second": 50.0,
    "time_to_first_text_seconds": null,
    "timing_scope": "server_request_wall_time_including_queue_and_generation"
  }
}
```

数值仅为字段示例。`inference_seconds` 从请求进入中间件开始，到完整 JSON 或 SSE `[DONE]` 就绪为止，包含排队、请求准备和生成，流式发送还可能受到背压影响；模型已常驻，因此不包含加载模型的时间，也不能当作纯 GPU 执行时间。流式额外的 `choices=[]` 事件携带 `profiling`，位于 `[DONE]` 之前。`time_to_first_text_seconds` 在流式模式记录首个非空文本片段到达时间，可能与第一个模型 token 不完全一致；非流式为 `null`。

客户端将响应中的 `profiling` 保存为 `server_metrics`，同时独立测量 `client_metrics.end_to_end_seconds`，其中包含网络开销。每次请求的 token 数取自 `usage.completion_tokens`；`text_chunks` 只是 SSE 文本片段数。缺少 token 计数时吞吐为 `null`，不会用片段数或最大输出长度代替。

分组汇总计算成功请求的实际输出 token 总数除以整组墙钟时间，同时输出成功 req/s、失败率、成功请求延迟 p50/p95（nearest-rank）。失败请求仍占用整组时间，不会从统计中隐藏。每请求 tokens/s 不可直接相加作为总吞吐。未自动预热的首次请求可能含初始化开销，正式比较仍需按规划冻结数据、预热并重复测试。

本地非 HTTP 推理在原有 `timings` 中补充 `inference_seconds`、`output_tokens_per_second` 和 `requests_per_second`，按实际整个 batch 输出计算，模型加载时间独立记录在 `load_seconds`。流式检查仍要求正常 finish_reason 和 `[DONE]`，中途截断判定失败。

### 标准库 Python 客户端

无需额外安装 HTTP SDK。这里调用的是本机 vLLM 服务；`model` 填启动时的别名。

```python
import json
from urllib.request import Request, urlopen

payload = {
    "model": "local-model",
    "messages": [{"role": "user", "content": "请用三句话解释 prefill 和 decode。"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "seed": 42,
    "chat_template_kwargs": {"enable_thinking": False},
}
request = Request(
    "http://127.0.0.1:7081/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=180) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
print(result["usage"])
```

也可以使用 OpenAI Python SDK 连接同一服务，流式读取回答：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7081/v1", api_key="EMPTY")
stream = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "请解释 KV cache。"}],
    max_tokens=128, temperature=0.7, top_p=0.8, seed=42, stream=True,
    extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

`top_k`、`chat_template_kwargs` 是 vLLM 扩展，SDK 用 `extra_body` 传入。网络 chunk 不是精确的单个 GPU token，不能直接把 chunk 间隔当作真实 ITL。接口细节参考 [vLLM 0.11.0 HTTP 服务文档](https://docs.vllm.ai/en/v0.11.0/serving/openai_compatible_server.html)。

## 6. 两个库如何选择、遇到问题如何定位

| 学习/运行目标 | 入口 | 先观察什么 |
| --- | --- | --- |
| 理解 tokenizer、模板与输入/输出切片 | Transformers / tokenize-only | 渲染文本、input_ids、attention_mask、新增 token |
| 单条生成、模型结构与生成策略 | Transformers | dtype、device、GenerationConfig、EOS |
| 本地批量推理与量化权重 | vLLM LLM | 引擎初始化日志、KV 预算、实际 quantization backend |
| 常驻服务、HTTP 请求与流式输出 | vLLM API server | 服务就绪、请求参数、usage、错误与流式事件 |
| 正式 BF16/AWQ/官方 FP8/本地 FP8 性能比较 | 已实现的 benchmark 与固定 workload | 默认每组预热5次、正式50次、1轮；--rounds 可选；本脚本只用于先跑通 |

- 缺少 `vllm`：确认当前 Python 安装了 README 指定的 wheel；requirements 中若仍注释 vLLM 行，需要单独执行加速安装命令。
- 缺少 `accelerate`：Transformers 的 device_map 需要它，按项目 requirements 安装对应版本。
- OOM / KV cache 不足：先用短输入、短输出、并发 1 和 eager；检查其他 GPU 进程。vLLM 显存比例过低也可能容不下权重/KV，不能一律靠继续降低比例解决。
- 模型生成到上限但没有完整回答：检查 thinking 是否开启，增加输出预算；上限截断不是正常 EOS 结束。
- undefined symbol / CUDA 扩展导入失败：核对实际 torch、vLLM wheel 和 CUDA 构建；驱动兼容与扩展 ABI 兼容需分别验证。
- 无模板或模板不支持 thinking：基础模型使用 raw prompt；其他 chat 模型使用自身模板，thinking 保留 auto。
- 出错时加 `--debug`，保存完整日志、运行命令、JSON 与环境快照，不把“能导入”视为“模型已经验证通过”。

建议本周按顺序完成四件事：先查看一条请求的模板/token；再运行一次 Transformers 生成；然后用 vLLM 生成同一问题；最后启动 HTTP 服务并用 Python 发送一条请求。每一步都保留真实结果，再进入固定 workload 与正式基线。
