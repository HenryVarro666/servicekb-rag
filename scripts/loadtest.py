"""简易压测：并发请求 HTTP API，统计吞吐（RPS）、延迟分位数（p50 / p95 / p99）和错误。

为什么要自己压一遍？
    单个请求跑得快 ≠ 并发下跑得快。连接池、线程池、GPU 服务的批处理、LLM 的并发配额……
    这些瓶颈只有在并发时才会出现。上线前、改了部署配置后，都应该压一遍并记录结果。

用法::

    servicekb serve --port 8000 &                         # 先启动服务
    python scripts/loadtest.py --endpoint /v1/search --concurrency 16 --duration 20
    python scripts/loadtest.py --endpoint /v1/ask --concurrency 8 --api-key change-me --groups public,service

问题从评估集（eval/golden_qa.jsonl）里轮流取。注意：/v1/ask 接了真实大模型时会产生费用。
只用标准库 + httpx（项目已依赖），不需要安装 locust 等工具。
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--endpoint", default="/v1/search", choices=["/v1/search", "/v1/ask"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=20.0, help="压测时长（秒）")
    ap.add_argument("--warmup", type=float, default=2.0, help="预热时长（秒），不计入统计")
    ap.add_argument("--questions", type=Path, default=ROOT / "eval" / "golden_qa.jsonl")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--groups", default="public,service")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果（方便记录 / 对比）")
    args = ap.parse_args()

    questions = [json.loads(line)["question"] for line in args.questions.read_text("utf-8").splitlines() if line.strip()]
    headers = {"X-User-Groups": args.groups}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    lat: list[float] = []
    status: Counter[str] = Counter()
    lock = threading.Lock()
    t_start = time.perf_counter()
    t_measure = t_start + args.warmup
    t_end = t_measure + args.duration

    def worker(wid: int) -> None:
        i = wid
        with httpx.Client(base_url=args.url, headers=headers, timeout=args.timeout) as client:
            while (now := time.perf_counter()) < t_end:
                q = questions[i % len(questions)]
                i += args.concurrency
                t0 = time.perf_counter()
                try:
                    r = client.post(args.endpoint, json={"question": q})
                    code = str(r.status_code)
                except httpx.HTTPError as e:
                    code = type(e).__name__
                dt = (time.perf_counter() - t0) * 1000
                if now >= t_measure:
                    with lock:
                        status[code] += 1
                        if code == "200":
                            lat.append(dt)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(worker, range(args.concurrency)))

    total = sum(status.values())
    result = {
        "endpoint": args.endpoint, "concurrency": args.concurrency, "duration_s": args.duration,
        "requests": total, "ok": status.get("200", 0), "rps": round(total / args.duration, 1),
        "p50_ms": round(percentile(lat, 50), 1), "p95_ms": round(percentile(lat, 95), 1),
        "p99_ms": round(percentile(lat, 99), 1), "mean_ms": round(statistics.fmean(lat), 1) if lat else 0.0,
        "status": dict(status),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return
    print(f"{args.endpoint}  并发 {args.concurrency}  时长 {args.duration:.0f}s")
    print(f"  请求 {total}（成功 {result['ok']}）  吞吐 {result['rps']} req/s")
    print(f"  延迟 p50 {result['p50_ms']} ms  p95 {result['p95_ms']} ms  p99 {result['p99_ms']} ms")
    if set(status) - {"200"}:
        print(f"  状态码分布 {dict(status)}")


if __name__ == "__main__":
    main()
