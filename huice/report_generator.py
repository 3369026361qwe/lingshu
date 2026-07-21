"""回测报告生成器 — JSON/Markdown/HTML格式 + DSR/PSR 数据窥探防御。"""
import json
from datetime import datetime, timezone


class ReportGenerator:
    @staticmethod
    def to_json(report: dict, path: str) -> str:
        """保存JSON报告。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        return path

    @staticmethod
    def to_markdown(report: dict) -> str:
        """生成Markdown报告字符串（含 DSR/PSR 数据窥探防御）。"""
        m = report.get("metrics", {})
        exp_id = report.get("experiment_id", "?")
        lines = [
            f"# 回测报告 — {exp_id}",
            f"生成时间: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## 绩效指标",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 累计收益率 | {m.get('total_return', 'N/A'):.2%} |" if isinstance(m.get('total_return'), float) else f"| 累计收益率 | {m.get('total_return', 'N/A')} |",
            f"| 年化收益率 | {m.get('annualized_return', 'N/A')} |",
            f"| 年化波动率 | {m.get('annualized_vol', 'N/A')} |",
            f"| 夏普比率 | {m.get('sharpe', 'N/A')} |",
            f"| 最大回撤 | {m.get('max_drawdown', 'N/A')} |",
            f"| 胜率 | {m.get('win_rate', 'N/A')} |",
            f"| 交易天数 | {m.get('n_days', 'N/A')} |",
        ]

        # DSR/PSR 数据窥探防御
        dsr_section = ReportGenerator._format_dsr_section(report)
        if dsr_section:
            lines.extend(dsr_section)

        lines.extend([
            "",
            "## 配置",
            "```json",
            json.dumps({k: str(v) for k, v in report.get("config", {}).items() if k not in ("data_loader", "signal_generator", "executor")}, ensure_ascii=False, indent=2),
            "```",
        ])
        return "\n".join(lines)

    @staticmethod
    def _format_dsr_section(report: dict) -> list[str]:
        """生成 DSR/PSR 数据窥探防御章节。

        如果 report 中包含 dsr_pvalue / haircut_sharpe / psr，
        则格式化为 markdown 表格；否则从原始数据中计算。
        """
        dsr_pvalue = report.get("dsr_pvalue")
        haircut_sr = report.get("haircut_sharpe")
        psr = report.get("psr")
        dsr_warning = report.get("dsr_warning")

        # 如果 report 中已有预计算值，直接使用
        has_dsr = dsr_pvalue is not None
        has_psr = psr is not None

        # 否则尝试从 metrics 和 config 计算
        if not has_dsr and not has_psr:
            m = report.get("metrics", {})
            sharpe = m.get("sharpe")
            n_days = m.get("n_days", 0)
            if sharpe is None or n_days < 20:
                return []

            n_trials = report.get("grid_search_trials", 1)
            cfg = report.get("config", {})
            if isinstance(cfg, dict) and "grid_params" in report:
                n_trials = max(n_trials, 2)  # 网格搜索中至少有2次试验

            from huice.data_snooping import DataSnoopingDefender

            dsr_pvalue = DataSnoopingDefender.deflated_sharpe_ratio(
                float(sharpe), n_trials, n_days,
            )
            haircut_sr = DataSnoopingDefender.haircut_sharpe(
                float(sharpe), n_trials, n_days,
            )
            psr = DataSnoopingDefender.probabilistic_sharpe_ratio(
                float(sharpe), 0.0, n_days,
            )
            has_dsr = True
            has_psr = True

        if not has_dsr and not has_psr:
            return []

        lines = ["", "## 数据窥探防御"]
        if dsr_warning:
            lines.append(f"**{dsr_warning}**")
            lines.append("")

        lines.extend([
            "| 指标 | 数值 | 说明 |",
            "|------|------|------|",
        ])

        if has_dsr:
            dsr_fmt = f"{dsr_pvalue:.4f}" if isinstance(dsr_pvalue, (int, float)) else str(dsr_pvalue)
            warning = " ⚠️ 可能存在数据窥探" if (
                isinstance(dsr_pvalue, (int, float)) and dsr_pvalue > 0.05
            ) else ""
            lines.append(f"| DSR p-value | {dsr_fmt}{warning} | 值越小越可靠，>0.05 需审查 |")
            if haircut_sr is not None:
                hsr_fmt = f"{haircut_sr:.4f}" if isinstance(haircut_sr, (int, float)) else str(haircut_sr)
                lines.append(f"| Haircut SR | {hsr_fmt} | 扣除多次试验偏差后的夏普 |")

        if has_psr:
            psr_fmt = f"{psr:.4f}" if isinstance(psr, (int, float)) else str(psr)
            psr_verdict = " ✓ 显著优于零" if (
                isinstance(psr, (int, float)) and psr > 0.95
            ) else ""
            lines.append(f"| PSR | {psr_fmt}{psr_verdict} | Prob(SR > 0)，>0.95 为显著 |")

        return lines

    @staticmethod
    def compute_dsr_psr(
        report: dict,
        n_trials: int = 1,
        benchmark_sharpe: float = 0.0,
    ) -> dict:
        """为报告补充 DSR/PSR 字段（就地修改 + 返回）。

        Args:
            report: 回测报告
            n_trials: 尝试的策略/参数组合数
            benchmark_sharpe: 基准夏普比率

        Returns:
            修改后的 report，新增 dsr_pvalue, haircut_sharpe, psr, dsr_warning
        """
        from huice.data_snooping import DataSnoopingDefender

        m = report.get("metrics", {})
        sharpe = m.get("sharpe")
        n_days = m.get("n_days", 0)

        if sharpe is None or n_days < 3:
            return report

        dsr_pvalue = DataSnoopingDefender.deflated_sharpe_ratio(
            float(sharpe), max(n_trials, 1), n_days,
        )
        haircut_sr = DataSnoopingDefender.haircut_sharpe(
            float(sharpe), max(n_trials, 1), n_days,
        )
        psr = DataSnoopingDefender.probabilistic_sharpe_ratio(
            float(sharpe), benchmark_sharpe, n_days,
        )

        report["dsr_pvalue"] = round(dsr_pvalue, 6)
        report["haircut_sharpe"] = round(haircut_sr, 4)
        report["psr"] = round(psr, 4)

        if dsr_pvalue > 0.05:
            report["dsr_warning"] = "⚠️ 可能存在数据窥探"

        return report

    @staticmethod
    def comparison_table(reports: list[dict]) -> str:
        """多实验对比Markdown表格。"""
        lines = ["| 实验ID | 夏普 | 累计收益 | 最大回撤 | 胜率 | DSR p-value |", "|------|------|------|------|------|------|"]
        for r in reports:
            eid = r.get("experiment_id", "?")[:20]
            m = r.get("metrics", {})
            dsr = r.get("dsr_pvalue", "?")
            dsr_fmt = f"{dsr:.3f}" if isinstance(dsr, float) else str(dsr)
            lines.append(f"| {eid} | {m.get('sharpe','?')} | {m.get('total_return','?')} | {m.get('max_drawdown','?')} | {m.get('win_rate','?')} | {dsr_fmt} |")
        return "\n".join(lines)
