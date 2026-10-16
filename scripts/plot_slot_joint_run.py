"""Plot saved v7 training logs without rescoring policies or changing a run.

Training updates and final-policy physical slots are deliberately separate.
Missing metrics stay missing; in particular a disabled KL diagnostic is not zero KL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


BLUE, TEAL, ORANGE, RED = "#2367a6", "#087f7b", "#df862e", "#be454b"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def series(history, key):
    return np.asarray([np.nan if row.get(key) is None else float(row[key]) for row in history])


def moving_mean(values, width):
    if width < 1:
        raise ValueError("Moving-average window must be positive")
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    for i, value in enumerate(values):
        if np.isfinite(value):
            window = values[max(0, i - width + 1):i + 1]
            result[i] = np.mean(window[np.isfinite(window)])
    return result


def load_run(run):
    manifest = read_json(run / "manifest.json")
    history = read_json(run / "history.json")
    evaluation = read_json(run / "evaluation.json")
    if manifest.get("protocol") != "slot_joint_v7" or not history:
        raise ValueError("A nonempty slot_joint_v7 run is required")
    if [row["update"] for row in history] != list(range(len(history))):
        raise ValueError("Training updates must be contiguous, unique, and start at zero")
    for key in ("protocol", "trajectory_digest", "warmup_deployment_digest", "evaluation_protocol"):
        if manifest.get(key) != evaluation.get(key):
            raise ValueError(f"Final evaluation provenance mismatch: {key}")
    expected = sum(count for _, count in manifest["training_settings"]["phase_schedule"])
    if expected != len(history) or len(evaluation["slots"]) != manifest["slots"]:
        raise ValueError("A completed training budget and its complete final evaluation are required")
    if [row["slot"] for row in evaluation["slots"]] != list(range(manifest["slots"])):
        raise ValueError("Final physical slots are not contiguous")
    if any(row.get("episode_length", manifest["slots"]) != manifest["slots"] for row in history):
        raise ValueError("Per-update slot averages require one complete trajectory per update")
    return manifest, history, evaluation


def metric_inventory(history, manifest):
    result = {}
    for key in sorted(set().union(*(row.keys() for row in history))):
        raw = [row[key] for row in history if row.get(key) is not None]
        if not raw or not all(isinstance(v, (float, int)) for v in raw):
            result[key] = {"status": "categorical", "values": sorted(set(map(str, raw)))}
            continue
        values = series(history, key)
        finite = values[np.isfinite(values)]
        status = "recorded"
        if key == "last_approx_kl" and manifest["ppo_config"].get("target_kl") is None:
            status = "placeholder_not_measured"
        result[key] = {"status": status, "finite_count": int(len(finite)),
                       "missing_count": int(len(values) - len(finite)),
                       "min": float(finite.min()) if len(finite) else None,
                       "max": float(finite.max()) if len(finite) else None}
    return result


def plot_run(run: Path, output: Path, window=20, dpi=165):
    if output.exists():
        raise FileExistsError("Choose a new figure directory; existing artifacts are not overwritten")
    manifest, history, evaluation = load_run(run)
    inventory = metric_inventory(history, manifest)
    attainment_target = manifest["objective"].get("attainment_target")
    names = {entry.name for entry in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                                             if n in names), "DejaVu Sans"),
                         "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    output.mkdir(parents=True)
    updates = series(history, "update") + 1
    boundaries = [event["checkpoint_next_update"] for event in manifest.get("training_budget_extensions", [])]
    figures = []

    def lines(ax, values, title, ylabel, *, color=BLUE, label=None, x=None):
        x = updates if x is None else x
        ax.plot(x, values, color=color, lw=.6, alpha=.23)
        ax.plot(x, moving_mean(values, window), color=color, lw=1.8,
                label=label or f"{window}轮滑动均值")
        ax.set(title=title, ylabel=ylabel, xlabel="PPO训练轮次")
        ax.grid(axis="y", alpha=.2)
        for boundary in boundaries:
            if x.min() < boundary < x.max():
                ax.axvline(boundary, color="#7f8996", ls="--", lw=.8, alpha=.7)

    def save(fig, name, title, note):
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.text(.01, .01, note, fontsize=9, color="#536171")
        fig.tight_layout(rect=(0, .045, 1, .945))
        fig.savefig(output / name, dpi=dpi)
        plt.close(fig)
        figures.append(name)

    def grid(rows=2, cols=2):
        return plt.subplots(rows, cols, figsize=(12.8, 3.45 * rows))

    fig, axes = grid()
    for ax, key, title, unit, color in [
        (axes[0, 0], "mean_utility", "绝对系统效用（越高越好）", "效用", BLUE),
        (axes[0, 1], "episode_cumulative_latency", "完成请求平均时延（越低越好）", "秒/请求", ORANGE),
        (axes[1, 0], "episode_cumulative_cost", "平均成本率（越低越好）", "相对货币/秒", TEAL),
        (axes[1, 1], "mean_constraint_attainment", "SLO达标率缺口（不是达标率）", "平均正违约量", RED),
    ]:
        values = series(history, key)
        if key.startswith("episode_cumulative_"):
            values = values / series(history, "episode_length")
        lines(ax, values, title, unit, color=color)
    axes[1, 0].set_ylim(bottom=0, top=max(series(history, "episode_cumulative_cost")
                                      / series(history, "episode_length")) * 1.1)
    axes[0, 0].legend(frameon=False)
    save(fig, "01_training_metrics.png", "1000轮模型选择训练：性能变化".replace("1000", str(len(history))),
         "浅色为逐轮原值，深色为滑动均值；灰虚线为预算延长点（未重新初始化）。Goodput、质量和原始达标率未逐轮记录。")

    fig, axes = grid(2, 3)
    for ax, key, title, color in [
        (axes[0, 0], "mean_learning_utility", "同槽效用差分（含首轮固定参照）", BLUE),
        (axes[0, 1], "mean_composition_lagrangian_reward", "每槽惩罚后奖励", RED),
        (axes[0, 2], "mean_reward", "每个子动作平均奖励", ORANGE),
        (axes[1, 0], "lagrange_attainment", "SLO约束乘子", RED),
    ]:
        lines(ax, series(history, key), title, "日志原值", color=color)
    axes[0, 0].set_yscale("symlog", linthresh=.001)
    axes[0, 0].set_ylabel("差分效用（对称对数轴）")
    for i, name in enumerate(("llm", "service", "network")):
        color = (BLUE, TEAL, ORANGE)[i]
        lines(axes[1, 1], series(history, f"mean_constraint_{name}"), "三类容量约束违约量", "违约量", color=color, label=name)
        lines(axes[1, 2], series(history, f"lagrange_{name}"), "三类容量约束乘子", "乘子", color=color, label=name)
    axes[1, 1].legend(frameon=False); axes[1, 2].legend(frameon=False)
    save(fig, "02_rewards_constraints.png", "奖励与约束：区分性能改进和惩罚变化",
         "SLO缺口使用当前绝对违约；效用使用同槽差分。惩罚乘子变化时，奖励曲线不是固定尺度的性能曲线。")

    slots = evaluation["slots"]
    x = np.asarray([row["slot"] + 1 for row in slots])
    fig, axes = grid(2, 3)
    for ax, key, title, unit, color in [
        (axes[0, 0], "mean_latency_s", "完成请求平均时延", "秒/请求", ORANGE),
        (axes[0, 1], "cost", "成本率", "相对货币/秒", TEAL),
        (axes[0, 2], "goodput_rps", "请求级Goodput", "SLO达标请求/秒", BLUE),
        (axes[1, 0], "quality", "请求级质量", "质量分", TEAL),
        (axes[1, 1], "slo_attainment", "SLO达标率", "%", RED),
    ]:
        values = np.asarray([row[key] for row in slots]) * (100 if key == "slo_attainment" else 1)
        ax.plot(x, values, "o-", color=color, lw=1.4, ms=3)
        ax.set(title=title, ylabel=unit, xlabel="物理时隙（1起始）")
        ax.grid(axis="y", alpha=.2)
    if attainment_target is not None:
        axes[1, 1].axhline(attainment_target * 100, color=RED, ls="--", label=f"{attainment_target:.0%}目标")
        axes[1, 1].legend(frameon=False)
    axes[1, 1].set_ylim(0, 105)
    completion = np.asarray([row["completed_rps"] / row["total_arrival_rps"]
                             if row["total_arrival_rps"] > 0 else np.nan for row in slots]) * 100
    axes[1, 2].plot(x, completion, "o-", color=BLUE, ms=3)
    axes[1, 2].set(title="请求完成率（不是SLO达标率）", ylabel="%", xlabel="物理时隙（1起始）", ylim=(0, 105))
    axes[1, 2].grid(axis="y", alpha=.2)
    save(fig, "03_final_policy_slots.png", f"第{len(history)}轮策略：原训练轨迹的{len(slots)}槽评价",
         "此图横轴是物理时隙，不是训练轮次；使用最终确定性策略，不是每轮随机采样策略的训练曲线。")

    fig, axes = grid()
    lines(axes[0, 0], series(history, "mean_loss"), "PPO总损失（完整范围）", "损失")
    axes[0, 0].set_yscale("symlog", linthresh=.001)
    mask = updates > 100 if len(updates) > 100 else np.ones(len(updates), dtype=bool)
    lines(axes[0, 1], series(history, "mean_loss")[mask], "PPO总损失（续训段，线性轴）", "损失", x=updates[mask])
    for key, color, label in [("routing_steps", BLUE, "模型选择子动作"), ("deployment_steps", ORANGE, "部署子动作")]:
        lines(axes[1, 0], series(history, key), "每轮决策数", "子动作数", color=color, label=label)
    axes[1, 0].legend(frameon=False)
    if manifest["ppo_config"].get("target_kl") is None:
        axes[1, 1].axis("off")
        axes[1, 1].text(.03, .85, "未测量或未保存的PPO诊断", fontsize=12, weight="bold")
        axes[1, 1].text(.03, .63, "target_kl=None时，日志中的KL=0只是占位值。\n不能据此判断策略没有更新。\n\nActor损失、Critic损失、熵、梯度范数\n和解释方差未分别保存在本次逐轮日志中。", va="top", linespacing=1.8)
    else:
        lines(axes[1, 1], series(history, "last_approx_kl"), "近似KL（启用测量）", "KL")
    save(fig, "04_ppo_diagnostics.png", "PPO诊断：损失与动作数量", "总损失不是效用；未记录的诊断不补造为零曲线。")

    fig, axes = grid(2, 3)
    groups = [
        ("每轮训练耗时", [("collection_time_s", "采样/环境"), ("optimization_time_s", "PPO更新")]),
        ("评价主要耗时", [("timing_analytical_s", "解析性能"), ("timing_workflow_s", "工作流")]),
        ("策略与观测耗时", [("timing_policy_act_s", "策略动作"), ("timing_observation_s", "观测")]),
        ("路由与检查耗时", [("timing_physical_routing_s", "物理路由"), ("timing_validation_s", "合法性检查")]),
        ("解析性能分项", [("timing_analytical_link_loads_s", "链路负载"), ("timing_analytical_node_distributions_s", "位置分布"), ("timing_analytical_llm_performance_s", "LLM性能")]),
        ("工作流分项", [(f"timing_workflow_{name}_s", label) for name, label in [("sampling", "采样"), ("assembly", "装配"), ("numeric", "数值计算"), ("aggregation", "汇总")]]),
    ]
    for ax, (title, entries) in zip(axes.flat, groups):
        for (key, label), color in zip(entries, (BLUE, ORANGE, TEAL, RED)):
            lines(ax, series(history, key), title, "秒/轮", color=color, label=label)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_yscale("log")
    save(fig, "05_runtime.png", "训练和评价耗时", "只绘制已有计时；后期新增的分项在早期留空，不用0填补。左上使用对数轴。")

    apps = sorted(set().union(*(row["app_latency_s"] for row in slots)))
    matrix = np.asarray([[row["app_latency_s"].get(app, np.nan) for row in slots] for app in apps])
    order = np.argsort(np.nanmean(matrix, axis=1))[::-1]
    fig, ax = plt.subplots(figsize=(13, 7.8))
    image = ax.imshow(matrix[order], aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(apps)), [apps[i] for i in order], fontsize=8)
    ax.set_xticks(range(0, len(slots), 2), x[::2])
    ax.set(xlabel="物理时隙（1起始）", ylabel="应用（按平均时延排序）")
    fig.colorbar(image, ax=ax, label="完成请求平均时延 / 秒")
    save(fig, "06_application_latency.png", f"最终策略：逐应用、逐槽完成请求时延",
         "这是最终策略的训练轨迹重放；未完成请求不计为0时延。缺失应用数据保持为空。")

    numeric = [key for key, row in inventory.items() if row["status"] != "categorical" and key != "update"]
    for page, start in enumerate(range(0, len(numeric), 8), 1):
        fig, axes = grid(4, 2)
        for ax, key in zip(axes.flat, numeric[start:start + 8]):
            if inventory[key]["status"] == "placeholder_not_measured":
                ax.axis("off"); ax.text(.05, .5, f"{key}\n未启用测量；0是占位值", va="center")
            else:
                lines(ax, series(history, key), textwrap.fill(key, 46), "日志原值")
                if inventory[key]["missing_count"]:
                    ax.text(.02, .95, f"缺失{inventory[key]['missing_count']}轮，留空", transform=ax.transAxes, va="top", fontsize=8)
        for ax in list(axes.flat)[len(numeric[start:start + 8]):]:
            ax.axis("off")
        save(fig, f"all_fields_{page:02}.png", f"完整日志字段 · 第{page}页", "保留原始字段、恒定值及缺失区间；重复口径也逐项展示，不将占位诊断视为测量值。")

    hashes = {name: hashlib.sha256((run / name).read_bytes()).hexdigest()
              for name in ("history.json", "manifest.json", "evaluation.json")}
    report = {"run": str(run.resolve()), "source_sha256": hashes, "updates": len(history),
              "physical_slots": len(slots), "moving_mean_window": window,
              "not_logged_per_update": ["goodput_rps", "quality", "slo_attainment"],
              "metric_inventory": inventory, "figures": figures}
    (output / "plot_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    index = ["## Material Passport", "", "- ARS experiment-agent；仅可视化现有结果，不重新训练或评分。",
             f"- 来源：`{run.resolve()}`；{len(history)}轮，{len(slots)}槽，单种子。",
             f"- 浅线为原值，深线为{window}轮滑动均值。末轮逐槽图不等于训练曲线。",
             "- Goodput、质量、原始SLO达标率未逐轮保存；KL测量未启用，不补造。", "",
             "# 训练图集", ""]
    for name in figures:
        index.extend([f"## {name}", "", f"![{name}]({name})", ""])
    (output / "README.md").write_text("\n".join(index), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    report = plot_run(args.run, args.output, args.window)
    print(f"Saved {len(report['figures'])} figures to {args.output}; {report['updates']} updates")


if __name__ == "__main__":
    main()
