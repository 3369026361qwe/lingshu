"""回测报告生成器 — JSON/Markdown/HTML格式。"""
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
        """生成Markdown报告字符串。"""
        m = report.get("metrics", {})
        exp_id = report.get("experiment_id", "?")
        lines = [
            f"# 回测报告 — {exp_id}",
            f"生成时间: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## 绩效指标",
            f"| 指标 | 数值 |",
            f"|------|------|",
            f"| 累计收益率 | {m.get('total_return', 'N/A'):.2%} |" if isinstance(m.get('total_return'), float) else f"| 累计收益率 | {m.get('total_return', 'N/A')} |",
            f"| 年化收益率 | {m.get('annualized_return', 'N/A')} |",
            f"| 年化波动率 | {m.get('annualized_vol', 'N/A')} |",
            f"| 夏普比率 | {m.get('sharpe', 'N/A')} |",
            f"| 最大回撤 | {m.get('max_drawdown', 'N/A')} |",
            f"| 胜率 | {m.get('win_rate', 'N/A')} |",
            f"| 交易天数 | {m.get('n_days', 'N/A')} |",
            "",
            "## 配置",
            f"```json",
            json.dumps({k: str(v) for k, v in report.get("config", {}).items() if k not in ("data_loader", "signal_generator", "executor")}, ensure_ascii=False, indent=2),
            f"```",
        ]
        return "\n".join(lines)

    @staticmethod
    def comparison_table(reports: list[dict]) -> str:
        """多实验对比Markdown表格。"""
        lines = ["| 实验ID | 夏普 | 累计收益 | 最大回撤 | 胜率 |", "|------|------|------|------|------|"]
        for r in reports:
            eid = r.get("experiment_id", "?")[:20]
            m = r.get("metrics", {})
            lines.append(f"| {eid} | {m.get('sharpe','?')} | {m.get('total_return','?')} | {m.get('max_drawdown','?')} | {m.get('win_rate','?')} |")
        return "\n".join(lines)
