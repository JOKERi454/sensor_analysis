#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据智能分析助手
======================
功能：
  1. 使用 Pandas 读取/清洗传感器数据（温度、湿度、光照、电压）
  2. 多维度异常检测（Z-Score、IQR、趋势异常）
  3. 生成结构化分析摘要
  4. 调用 DeepSeek API 生成中文可读性强的分析报告
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import requests
from typing import Optional

# ============================================================
# 0. 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "dfae3-main", "传感器数据集", "传感器数据集", "data完整.txt")
POS_FILE  = os.path.join(BASE_DIR, "dfae3-main", "传感器数据集", "传感器数据集", "位置信息.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# DeepSeek API 配置
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"   # DeepSeek-V3

# 列名（原始 txt 无表头）
COLUMN_NAMES = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]

# 异常检测参数
ZSCORE_THRESHOLD   = 3.0   # Z-Score 阈值
IQR_MULTIPLIER     = 1.5   # IQR 倍数
MIN_RECORDS_PER_NODE = 50  # 每个节点最少记录数才纳入节点级分析


# ============================================================
# 1. 数据读取与预处理
# ============================================================
def load_sensor_data(data_path: str, chunk_size: int = 500_000) -> pd.DataFrame:
    """分块读取大型传感器数据文件，返回完整 DataFrame"""
    print(f"正在读取数据文件: {data_path}")
    t0 = time.time()

    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(
        data_path,
        sep=r"\s+",
        names=COLUMN_NAMES,
        header=None,
        dtype={
            "date": str, "time": str, "epoch": "int64",
            "moteid": "int16", "temperature": "float32",
            "humidity": "float32", "light": "float32", "voltage": "float32"
        },
        chunksize=chunk_size,
        engine="python",
    ):
        # 合并 date + time -> datetime
        chunk["datetime"] = pd.to_datetime(
            chunk["date"] + " " + chunk["time"], errors="coerce"
        )
        # 过滤明显异常的温度值（传感器故障值，如 122°C）
        chunk = chunk[
            (chunk["temperature"].between(-50, 80)) &
            (chunk["humidity"].between(0, 110)) &
            (chunk["light"].between(0, 2000)) &
            (chunk["voltage"].between(0, 5))
        ]
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"  已读取 {total_rows:>10,} 行 ...")

    df = pd.concat(chunks, ignore_index=True)
    print(f"数据读取完成: {len(df):,} 行有效数据, 耗时 {time.time()-t0:.1f}s")
    return df


def load_position_data(pos_path: str) -> pd.DataFrame:
    """读取传感器位置信息（跳过末尾的 schema 描述行）"""
    print(f"正在读取位置信息: {pos_path}")
    # 先读取所有行，过滤掉包含冒号的行（schema 描述）
    with open(pos_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and ":" not in line]
    # 写入临时文件再读取（pandas 处理字符串行比较方便）
    from io import StringIO
    pos_df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
        names=["moteid", "x", "y"],
        header=None,
        dtype={"moteid": "int16", "x": "float32", "y": "float32"},
    )
    print(f"位置信息: {len(pos_df)} 个节点")
    return pos_df


# ============================================================
# 2. 异常检测
# ============================================================
def detect_anomalies_zscore(series: pd.Series, threshold: float = ZSCORE_THRESHOLD) -> pd.Series:
    """Z-Score 异常检测（适用于近似正态分布的数据）"""
    z = np.abs(stats.zscore(series, nan_policy="omit"))
    return z > threshold


def detect_anomalies_iqr(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> pd.Series:
    """IQR 异常检测（对偏态分布更鲁棒）"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    return (series < lower) | (series > upper)


def detect_anomalies_rolling(
    series: pd.Series, window: int = 100, n_std: float = 3.0
) -> pd.Series:
    """滑动窗口异常检测：偏离局部均值超过 n_std 倍标准差"""
    rolling_mean = series.rolling(window=window, center=True, min_periods=10).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=10).std()
    upper = rolling_mean + n_std * rolling_std
    lower = rolling_mean - n_std * rolling_std
    return (series > upper) | (series < lower)


def comprehensive_anomaly_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    综合异常分析：
      - Z-Score >= 3
      - IQR 超出 1.5 倍
      - 滑动窗口局部异常
    对 temperature, humidity, light, voltage 分别检测
    """
    print("\n ========== 开始异常检测 ==========")
    t0 = time.time()

    metrics = ["temperature", "humidity", "light", "voltage"]
    anomaly_flags = pd.DataFrame(index=df.index)

    for metric in metrics:
        print(f"  检测指标: {metric} ...")
        series = df[metric]

        # 1) Z-Score
        anz = detect_anomalies_zscore(series)
        # 2) IQR
        aiq = detect_anomalies_iqr(series)
        # 3) 滑动窗口
        aro = detect_anomalies_rolling(series, window=200, n_std=3.0)

        # 综合：至少两种方法都判定为异常
        combined = (anz.astype(int) + aiq.astype(int) + aro.astype(int)) >= 2
        anomaly_flags[f"{metric}_anomaly"] = combined
        anomaly_flags[f"{metric}_zscore"]   = anz
        anomaly_flags[f"{metric}_iqr"]      = aiq
        anomaly_flags[f"{metric}_rolling"]  = aro

        n_anom = combined.sum()
        pct = 100 * n_anom / len(df)
        print(f"     -> {metric}: {n_anom:,} 条异常 ({pct:.2f}%)")

    # 整体异常标记
    anomaly_flags["is_anomaly"] = anomaly_flags[
        [f"{m}_anomaly" for m in metrics]
    ].any(axis=1)

    result = pd.concat([df, anomaly_flags], axis=1)
    total_anomalies = result["is_anomaly"].sum()
    print(f"异常检测完成: 总计 {total_anomalies:,} 条异常记录 "
          f"({100*total_anomalies/len(df):.2f}%), 耗时 {time.time()-t0:.1f}s")
    return result


# ============================================================
# 3. 统计分析 & 摘要生成
# ============================================================
def generate_analysis_summary(df: pd.DataFrame, pos_df: pd.DataFrame) -> dict:
    """生成结构化分析摘要，用于构造 prompt"""
    print(" ========== 生成分析摘要 ==========")
    summary = {}

    # ---- 基础信息 ----
    summary["数据概览"] = {
        "总记录数": f"{len(df):,}",
        "传感器节点数": df["moteid"].nunique(),
        "时间范围": f"{df['datetime'].min()} 至 {df['datetime'].max()}",
        "数据时间跨度(天)": f"{(df['datetime'].max() - df['datetime'].min()).days}",
    }

    # ---- 各指标统计 ----
    metrics_stats = {}
    for metric in ["temperature", "humidity", "light", "voltage"]:
        s = df[metric]
        metrics_stats[metric] = {
            "均值":   round(s.mean(), 4),
            "标准差": round(s.std(), 4),
            "最小值": round(s.min(), 4),
            "25%分位": round(s.quantile(0.25), 4),
            "中位数": round(s.median(), 4),
            "75%分位": round(s.quantile(0.75), 4),
            "最大值": round(s.max(), 4),
        }
    summary["指标统计"] = metrics_stats

    # ---- 异常统计 ----
    anomaly_summary = {}
    for metric in ["temperature", "humidity", "light", "voltage"]:
        col = f"{metric}_anomaly"
        if col in df.columns:
            n = df[col].sum()
            anomaly_summary[metric] = {
                "异常记录数": int(n),
                "异常占比(%)": round(100 * n / len(df), 3),
                "Z-Score异常数": int(df[f"{metric}_zscore"].sum()),
                "IQR异常数": int(df[f"{metric}_iqr"].sum()),
                "滚动窗口异常数": int(df[f"{metric}_rolling"].sum()),
            }
    summary["异常统计"] = anomaly_summary

    total_anom = df["is_anomaly"].sum() if "is_anomaly" in df.columns else 0
    summary["综合异常"] = {
        "总异常记录数": int(total_anom),
        "总异常占比(%)": round(100 * total_anom / len(df), 3),
    }

    # ---- 按节点统计异常率 ----
    node_anomaly = df.groupby("moteid").agg(
        总记录数=("moteid", "count"),
        异常记录数=("is_anomaly", "sum") if "is_anomaly" in df.columns else ("temperature_anomaly", "sum"),
        温度均值=("temperature", "mean"),
        湿度均值=("humidity", "mean"),
        光照均值=("light", "mean"),
        电压均值=("voltage", "mean"),
        温度异常数=("temperature_anomaly", "sum") if "temperature_anomaly" in df.columns else ("temperature_zscore", "sum"),
        湿度异常数=("humidity_anomaly", "sum") if "humidity_anomaly" in df.columns else ("humidity_zscore", "sum"),
        光照异常数=("light_anomaly", "sum") if "light_anomaly" in df.columns else ("light_zscore", "sum"),
        电压异常数=("voltage_anomaly", "sum") if "voltage_anomaly" in df.columns else ("voltage_zscore", "sum"),
    ).reset_index()

    node_anomaly["异常率(%)"] = round(
        100 * node_anomaly["异常记录数"] / node_anomaly["总记录数"], 2
    )

    # 合并位置信息
    if pos_df is not None:
        node_anomaly = node_anomaly.merge(pos_df, on="moteid", how="left")

    # top-5 异常率最高的节点
    top_anomaly_nodes = node_anomaly.nlargest(5, "异常率(%)")
    summary["异常率最高节点TOP5"] = top_anomaly_nodes.to_dict(orient="records")

    # ---- 按时间聚合异常趋势 ----
    if "datetime" in df.columns:
        df_time = df.set_index("datetime")
        daily_anomaly = df_time.resample("1d").agg(
            总记录数=("is_anomaly", "count"),
            异常记录数=("is_anomaly", "sum"),
            温度均值=("temperature", "mean"),
            湿度均值=("humidity", "mean"),
            光照均值=("light", "mean"),
            电压均值=("voltage", "mean"),
        ).dropna()

        daily_anomaly["异常率(%)"] = round(
            100 * daily_anomaly["异常记录数"] / daily_anomaly["总记录数"], 2
        )
        # 异常率最高的5天
        top_anomaly_days = daily_anomaly.nlargest(5, "异常率(%)")
        summary["异常率最高日期TOP5"] = top_anomaly_days.reset_index().to_dict(orient="records")

        # 趋势摘要
        summary["时间趋势"] = {
            "异常率趋势": "上升" if daily_anomaly["异常率(%)"].iloc[-1] > daily_anomaly["异常率(%)"].iloc[0] else "下降",
            "平均每日异常率(%)": round(daily_anomaly["异常率(%)"].mean(), 2),
            "最高异常率(%)": round(daily_anomaly["异常率(%)"].max(), 2),
            "最高异常率日期": str(daily_anomaly["异常率(%)"].idxmax().date()),
        }

    # ---- 节点位置聚类（简单分组） ----
    if pos_df is not None:
        merged = node_anomaly.dropna(subset=["x", "y"])
        if len(merged) >= 10:
            # 按位置大致分为4个区域
            x_med, y_med = merged["x"].median(), merged["y"].median()
            summary["空间分布"] = {
                "节点总数(有位置信息)": len(merged),
                "X坐标范围": f"({merged['x'].min():.1f}, {merged['x'].max():.1f})",
                "Y坐标范围": f"({merged['y'].min():.1f}, {merged['y'].max():.1f})",
                "各区域节点数": {
                    "左上": int(((merged["x"] <= x_med) & (merged["y"] > y_med)).sum()),
                    "右上": int(((merged["x"] > x_med) & (merged["y"] > y_med)).sum()),
                    "左下": int(((merged["x"] <= x_med) & (merged["y"] <= y_med)).sum()),
                    "右下": int(((merged["x"] > x_med) & (merged["y"] <= y_med)).sum()),
                }
            }

    return summary


def save_analysis_results(df: pd.DataFrame, summary: dict):
    """保存分析结果到 output 目录"""
    # 保存异常标记后的数据样本
    anomaly_sample = df[df["is_anomaly"]].head(10_000)
    sample_path = os.path.join(OUTPUT_DIR, "anomaly_samples.csv")
    anomaly_sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f" 异常样本已保存: {sample_path}")

    # 保存完整摘要 JSON
    summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"分析摘要已保存: {summary_path}")


# ============================================================
# 4. Prompt 构造
# ============================================================
def build_deepseek_prompt(summary: dict) -> str:
    """基于分析摘要构造发送给 DeepSeek 的 prompt"""
    prompt = f"""你是一位资深的物联网传感器数据分析专家。请根据以下传感器数据的分析摘要，用中文撰写一份全面、可读性强的分析报告。

## 分析摘要

### 一、数据概览
- 总记录数: {summary['数据概览']['总记录数']}
- 传感器节点数: {summary['数据概览']['传感器节点数']}
- 时间范围: {summary['数据概览']['时间范围']}
- 数据时间跨度: {summary['数据概览']['数据时间跨度(天)']}

### 二、各指标描述性统计
- 温度(°C): 均值={summary['指标统计']['temperature']['均值']}, 标准差={summary['指标统计']['temperature']['标准差']}, 范围=[{summary['指标统计']['temperature']['最小值']}, {summary['指标统计']['temperature']['最大值']}]
- 湿度(%): 均值={summary['指标统计']['humidity']['均值']}, 标准差={summary['指标统计']['humidity']['标准差']}, 范围=[{summary['指标统计']['humidity']['最小值']}, {summary['指标统计']['humidity']['最大值']}]
- 光照(lux): 均值={summary['指标统计']['light']['均值']}, 标准差={summary['指标统计']['light']['标准差']}, 范围=[{summary['指标统计']['light']['最小值']}, {summary['指标统计']['light']['最大值']}]
- 电压(V): 均值={summary['指标统计']['voltage']['均值']}, 标准差={summary['指标统计']['voltage']['标准差']}, 范围=[{summary['指标统计']['voltage']['最小值']}, {summary['指标统计']['voltage']['最大值']}]

### 三、异常检测结果
- 总异常记录数: {summary['综合异常']['总异常记录数']}，占比 {summary['综合异常']['总异常占比(%)']}%
- 温度异常: {summary['异常统计']['temperature']}
- 湿度异常: {summary['异常统计']['humidity']}
- 光照异常: {summary['异常统计']['light']}
- 电压异常: {summary['异常统计']['voltage']}

### 四、异常率最高节点TOP5
{json.dumps(summary['异常率最高节点TOP5'], ensure_ascii=False, indent=2, default=str)}

### 五、时间趋势
{json.dumps(summary.get('时间趋势', {}), ensure_ascii=False, indent=2, default=str)}

### 六、异常率最高日期TOP5
{json.dumps(summary.get('异常率最高日期TOP5', []), ensure_ascii=False, indent=2, default=str)}

### 七、空间分布
{json.dumps(summary.get('空间分布', {}), ensure_ascii=False, indent=2, default=str)}

## 撰写要求

请严格按照以下结构撰写报告：

1. **执行摘要 (200字以内)**：用精炼的语言概括最能说明问题的数据和发现。
2. **数据质量评估**：评价数据的完整性、一致性和可靠性，指出潜在的数据质量问题。
3. **各指标详细分析**：分别对温度、湿度、光照、电压四个指标进行分析，包括正常范围、波动情况、异常特征。
4. **异常模式深度解读**：
   - 分析高异常率节点的共同特征和可能原因
   - 分析异常的时间分布特征（是否集中出现在某些时段）
   - 分析异常的空间分布特征（是否存在区域性异常聚集）
5. **关联性分析**：分析四个指标之间的相关性，如温湿度关系、电压波动对传感器读数的影响等。
6. **根因推断**：基于数据特征，推断可能导致异常的原因（如传感器故障、环境变化、供电问题等）。
7. **结论与建议**：给出可操作的结论和维护建议。

## 格式要求
- 报告用中文撰写，语言流畅、专业但不晦涩。
- 使用 Markdown 格式组织，包含清晰的标题层级。
- 关键数字使用**粗体**突出。
- 适当使用列表和表格提高可读性。
- 总字数控制在 2000-3000 字。
"""
    return prompt


# ============================================================
# 5. DeepSeek API 调用
# ============================================================
def call_deepseek_api(prompt: str, api_key: str) -> Optional[str]:
    """调用 DeepSeek API 生成分析报告"""
    if api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
        print("\n[WARNING] 未设置 DEEPSEEK_API_KEY 环境变量！")
        print("  请通过以下方式设置:")
        print('    export DEEPSEEK_API_KEY="sk-xxxxxxxx"')
        print("  或者直接在脚本中修改 DEEPSEEK_API_KEY 变量")
        print("\n  下面将打印构造好的 prompt，并保存到文件，供手动使用。\n")
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一位专业的物联网传感器数据分析师，擅长用中文撰写清晰、深入的数据分析报告。请严格按照用户要求的格式进行回复。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 0.95,
    }

    print(f"\n 正在调用 DeepSeek API ({DEEPSEEK_MODEL}) ...")
    t0 = time.time()

    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        report = result["choices"][0]["message"]["content"]
        elapsed = time.time() - t0
        print(f"API 调用成功! 耗时 {elapsed:.1f}s, 返回 {len(report)} 字符")

        # 计算 token 用量
        usage = result.get("usage", {})
        print(f"Token 用量: prompt={usage.get('prompt_tokens','?')}, "
              f"completion={usage.get('completion_tokens','?')}, "
              f"total={usage.get('total_tokens','?')}")

        return report

    except requests.exceptions.Timeout:
        print("[ERROR] API 请求超时（120s）")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API 请求失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  响应内容: {e.response.text[:500]}")
        return None


# ============================================================
# 6. 报告保存
# ============================================================
def save_report(prompt: str, report: str):
    """保存 prompt 和报告到 output 目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 prompt
    prompt_path = os.path.join(OUTPUT_DIR, f"prompt_{timestamp}.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Prompt 已保存: {prompt_path}")

    # 保存报告
    if report:
        report_path = os.path.join(OUTPUT_DIR, f"analysis_report_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 传感器数据智能分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            f.write(report)
        print(f"分析报告已保存: {report_path}")
        return report_path
    return None


# ============================================================
# 7. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("  传感器数据智能分析助手")
    print("  Sensor Data Intelligent Analysis Assistant")
    print("=" * 60)

    # Step 1: 读取数据
    df = load_sensor_data(DATA_FILE)
    pos_df = load_position_data(POS_FILE)

    # Step 2: 异常检测
    df = comprehensive_anomaly_analysis(df)

    # Step 3: 生成分析摘要
    summary = generate_analysis_summary(df, pos_df)
    save_analysis_results(df, summary)

    # Step 4: 构造 Prompt
    print("\n ========== 构造 DeepSeek Prompt ==========")
    prompt = build_deepseek_prompt(summary)
    print(f" Prompt 构造完成, 长度: {len(prompt)} 字符")

    # Step 5: 调用 DeepSeek API
    report = call_deepseek_api(prompt, DEEPSEEK_API_KEY)

    # Step 6: 保存报告
    report_path = save_report(prompt, report)

    if report:
        print("\n" + "=" * 60)
        print("  分析报告预览 (前800字)")
        print("=" * 60)
        print(report[:800])
        print("...")
        print(f"\n完整报告请查看: {report_path}")

    print("\n 全流程完成!")
    return df, summary, report


if __name__ == "__main__":
    main()
