# Sensor Log Intelligent Analysis Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./dfae3-main/LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-✔-150458.svg)](https://pandas.pydata.org/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek--V3-536DFE.svg)](https://deepseek.com/)

基于 Python 的物联网传感器日志智能分析助手。对大规模无线传感器网络数据进行自动清洗、多维度异常检测，并调用 DeepSeek-V3 大模型自动生成专业的中文分析报告。

## ✨ 核心功能

- **海量数据处理** — 使用 Pandas 分块读取 144MB / 230 万行传感器数据，内存友好
- **智能数据清洗** — 基于物理边界自动过滤异常读数（温度、湿度、光照、电压）
- **多维度异常检测** — 三种方法综合投票，降低误报率
  - Z-Score（3σ 阈值）
  - IQR（四分位距法，鲁棒处理偏态分布）
  - 滚动窗口（局部异常捕获）
- **全维度统计分析** — 按指标、按节点、按时间维度聚合，含空间分布分析
- **AI 报告生成** — 调用 DeepSeek API 自动生成七部分结构化中文分析报告
- **多格式输出** — 异常样本 CSV + 结构化摘要 JSON + Markdown 分析报告

## 📊 数据集

数据来源于 MIT 计算机科学实验室的无线传感器网络研究项目（Intel Berkeley Research Lab）。

| 项目 | 详情 |
|------|------|
| 数据量 | 2,313,153 条记录（144 MB） |
| 传感器节点 | 55 个无线节点 |
| 时间跨度 | 2004-02-28 至 2004-04-05（37 天） |
| 采集指标 | 温度、湿度、光照、电压 |
| 采样频率 | 每 31 秒一次 |
| 附加信息 | 54 个节点的平面坐标位置 |

数据来源：<https://gitcode.com/open-source-toolkit/dfae3.git>

## 📁 项目结构

```
Sensor-Log-Intelligent-Analysis-Assistant/
├── sensor_analysis_pipeline.py          # 主流水线脚本
├── dfae3-main/
│   ├── LICENSE                          # MIT 许可证
│   ├── 传感器数据集.rar                  # 数据压缩包（32 MB）
│   └── 传感器数据集/
│       ├── data完整.txt                  # 主数据文件（144 MB）
│       ├── 位置信息.txt                  # 节点坐标
│       └── 具体参数网址.txt              # 数据来源
└── output/
    ├── anomaly_samples.csv              # 异常样本导出
    ├── analysis_summary.json            # 结构化分析摘要
    ├── analysis_report_[时间戳].md       # AI 生成报告
    └── prompt_[时间戳].md               # 发送给 API 的提示词
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 依赖库：`pandas` `numpy` `scipy` `requests`

### 安装依赖

```bash
pip install pandas numpy scipy requests
```

### 配置 API 密钥

设置 DeepSeek API 密钥环境变量：

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Windows (CMD)
set DEEPSEEK_API_KEY=your-deepseek-api-key

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your-deepseek-api-key"
```

> 也可直接修改 `sensor_analysis_pipeline.py` 第 34 行的 `DEEPSEEK_API_KEY`。

### 运行

```bash
python sensor_analysis_pipeline.py
```

执行流程约需 2-5 分钟（含 API 调用），完成后在 `output/` 目录查看结果。

## 🔬 检测方法详解

三种算法各有侧重，综合投票（至少两种方法同时判定）才标记为异常：

| 方法 | 原理 | 参数 | 适用场景 |
|------|------|------|----------|
| **Z-Score** | 偏离均值超过 N 个标准差 | N=3.0 | 近似正态分布指标（温度、湿度） |
| **IQR** | 超出 Q1-1.5×IQR 或 Q3+1.5×IQR | k=1.5 | 偏态分布指标（光照） |
| **滚动窗口** | 在 200 点窗口中偏离局部均值超过 3σ | w=200, σ=3.0 | 捕获局部渐变异常 |

### 物理边界过滤

| 指标 | 合理范围 |
|------|----------|
| 温度 | -50°C ~ 80°C |
| 湿度 | 0% ~ 110% |
| 光照 | 0 ~ 2000 lux |
| 电压 | 0 ~ 5V |

## 📈 分析结果示例

最近一次运行（1,835,134 条有效记录）：

| 指标 | 异常数 | 异常率 |
|------|--------|--------|
| 温度 | 25,670 | 1.40% |
| 湿度 | 12,996 | 0.71% |
| 光照 | 497 | 0.03% |
| 电压 | 4,409 | 0.24% |
| **合计** | **43,079** | **2.35%** |

### 关键发现

1. **供电/硬件故障** — 节点 55、56、58 电压异常率分别高达 44.87%、33.35%、20.32%，疑为电源模块老化或接触不良
2. **传感器老化** — 节点 21、25 温湿度同时异常，符合传感器元件漂移特征
3. **整体趋势向好** — 异常率随时间呈下降趋势

AI 生成的完整报告见 `output/analysis_report_[时间戳].md`。

## 📝 配置说明

所有可调参数在 `sensor_analysis_pipeline.py` 顶部集中管理：

```python
# 异常检测参数
ZSCORE_THRESHOLD   = 3.0    # Z-Score 阈值
IQR_MULTIPLIER     = 1.5    # IQR 倍数
MIN_RECORDS_PER_NODE = 50   # 节点最少记录数

# API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-key")
DEEPSEEK_MODEL   = "deepseek-chat"  # DeepSeek-V3
```

## 📄 许可

本项目代码和数据集均采用 [MIT 许可证](./dfae3-main/LICENSE)。数据集仅限研究用途。

## 🙏 致谢

- 数据来源：MIT Computer Science & Artificial Intelligence Lab — Intel Berkeley Research Lab
- 大模型支持：[DeepSeek](https://deepseek.com/)
- 核心依赖：[Pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/)
