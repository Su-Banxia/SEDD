# Score Entropy Discrete Diffusion（SEDD） 复现和微调

具体数学原理见 `SEDD.md`，个人理解可能存在很多错漏，欢迎指出

## 项目结构

```tex
Score-Entropy-Discrete-Diffusion/
├── configs/                        # Hydra 配置文件
│   ├── config.yaml                 #   主配置（训练、采样、数据、优化器等）
│   └── model/
│       ├── small.yaml              #   small 模型: 768h / 12blocks / 12heads
│       └── medium.yaml             #   medium 模型: 1024h / 24blocks / 16heads
├── model/                          # 模型定义
│   ├── __init__.py
│   ├── transformer.py              #   SEDD 主体（DDiT Block,Rotary, PyTorchModelHubMixin）
│   ├── rotary.py                   #   旋转位置编码 RoPE
│   ├── ema.py                      #   指数移动平均 EMA
│   ├── fused_add_dropout_scale.py  #   融合 add-dropout-scale 算子
│   └── utils.py                    #   get_score_fn（可切换 log-score / exp-score）
├── data/                           # 数据目录（gitignore）
│   └── alpaca/						# 以 Alpaca 数据集为例
│       ├── train.jsonl         
│       └── valid.jsonl           
├── sedd-small/                     # 本地缓存的预训练模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── model.safetensors
├── exp_local/                      # 实验输出目录（gitignore）
│   └── <data.train>/<date>/<time>/
│       ├── .hydra/                 #   Hydra 运行时配置快照
│       ├── checkpoints/            #   定期保存的检查点
│       ├── checkpoints-meta/       #   用于断点续训的元检查点
│       ├── samples/                #   训练过程中的采样结果
│       └── logs                    #   训练日志
├── comparison_results/             # 模型对比输出（gitignore）
│   └── <timestamp>/
│       ├── base_outputs.txt
│       ├── finetuned_outputs.txt
│       └── comparison_report.txt
│
├── noise_lib.py                    # 噪声调度：GeometricNoise / LogLinearNoise
├── graph_lib.py                    # 前向扩散图：Uniform / Absorbing
├── losses.py                       # 损失函数：score_entropy + get_step_fn
├── sampling.py                     # 采样策略：Euler / Analytic 预测器 + Denoiser
├── data.py                         # 数据加载：HF datasets + GPT2 tokenizer + DDP
├── train.py                        # 训练循环主要逻辑
├── load_model.py                   # 模型加载：HuggingFace Hub / 本地路径
├── utils.py                        # 工具：Logger, checkpoint save/restore
│
├── run_train.py                    # 入口：从零训练
├── run_finetune.py                 # 入口：基于预训练模型微调
├── run_sample.py                   # 入口：无条件采样
├── run_sample_cond.py              # 入口：条件采样
├── compare_models.py               # 入口：对比基座模型和微调模型
├── prepare_data.py                 # 工具：下载 Alpaca 并转为 jsonl
├── catsample.py                    # 工具：拼接采样输出为单文件
├── test.py                         # 测试脚本
│
├── environment.yml                 # Conda 环境定义
├── SEDD.md                  		# 数学原理详细文档
└── LICENSE                         # 原仓库 MIT License
```

---

## 环境配置与数据准备

快速安装：

```bash
git clone https://github.com/Su-Banxia/SEDD.git
cd Score-Entropy-Discrete-Diffusion
conda env create -f environment.yml
conda activate sedd
```

运行准备脚本：

```bash
python prepare_data.py
```

默认会下载并处理 `tatsu-lab/alpaca` 至 train.jsonl 与 valid.jsonl。格式为每行一个 JSON，字段 `text` 包含 "Instruction/ Input/ Output" 的拼接文本。

自定义数据：将数据格式化为相同 jsonl，每行 {"text": "..."}，放到 `data/<dataset>/`，并在运行时用命令行覆盖 `data.train`/`data.valid`。

---

## 运行

以下示例使用 4 x 4090D：

#### 模型训练

```bash
# 默认配置（absorb + loglinear + small）
python run_train.py ngpus=4

# uniform 图 + geometric 噪声
python run_train.py ngpus=4 noise.type=geometric graph.type=uniform model.scale_by_sigma=False

# 以 medium 模型为例
python run_train.py ngpus=4 model=medium training.accum=2
```

参数：
- `ngpus`：GPU 数量
- `training.batch_size`：每卡批大小
- `training.accum`：梯度累积步数
- `training.n_iters`：训练步数
- `optim.lr`：学习率
- `training.ema`：EMA 衰减率

训练输出在 `exp_local/<data.train>/<YYYY.MM.DD>/<HHMMSS>/`，包含 `.hydra/` 配置快照、`checkpoints/`、`samples/`、`logs/` 等。

#### 微调（以 Alpaca 为例）

此处采用全量微调

```bash
python run_finetune.py \
  ngpus=4 \
  pretrained_model_path=sedd-small \
  data.train=data/alpaca/train.jsonl \
  data.valid=data/alpaca/valid.jsonl \
  training.batch_size=128 \
  training.accum=4 \
  training.n_iters=10000 \
  optim.lr=1e-4
```

说明：
- `pretrained_model_path` 可为 HuggingFace Hub 路径（例如 `louaaron/sedd-small`）或本地模型目录；
- 微调时通常增大 batch（或累积步数）并减小学习率。

#### 采样

无条件与条件采样示例：

```bash
# 无条件采样
python run_sample.py --model_path sedd-small --steps 256

# 条件采样示例（prefix）
python run_sample_cond.py --model_path sedd-small --steps 256 --prefix "Instruction: Write a poem about AI.\nInput:\nOutput:"
```

#### 模型对比

项目提供 `compare_models.py`，会抽取 Alpaca 验证前缀，用基座模型与微调模型分别生成并保存对比结果：

```bash
python compare_models.py
# 输出位于 comparison_results/<timestamp>/ 包含 base_outputs.txt, finetuned_outputs.txt, comparison_report.txt，具体目录请手动调整
```

---

## 配置系统（Hydra）

默认配置文件位于 config.yaml，模型大小在 `configs/model/*.yaml` 中定义。所有配置项均可通过命令行覆盖，例如：

```bash
python run_train.py ngpus=4 training.batch_size=64 optim.lr=1e-4 model=medium
```
