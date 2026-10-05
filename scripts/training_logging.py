"""用独立监控进程记录训练输出、异常和退出状态，无需加载仿真依赖。"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import os
import shlex
import signal
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

RUN_DIRECTORY_ENV = "TA_SRU_LOGGED_RUN_DIRECTORY"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(directory: Path, **values) -> None:
    """原子替换状态文件，避免读取到未写完的 JSON。"""
    path = directory / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(values)
    temporary = directory / f"status.{os.getpid()}.json.tmp"
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def record_failure(directory: Path, error: BaseException) -> None:
    """保存原始异常；后续关闭或 checkpoint 保存失败不能覆盖它。"""
    try:
        path = directory / "error.json"
        if not path.exists():
            path.write_text(
                json.dumps(
                    {
                        "time": _now(),
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": "".join(
                            traceback.format_exception(
                                type(error), error, error.__traceback__
                            )
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        write_status(
            directory,
            state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
        )
    except Exception:  # noqa: BLE001
        # 日志失败只输出诊断，不能覆盖原始异常。
        traceback.print_exc()


class TrainingInterrupted(KeyboardInterrupt):
    """保存导致中断的信号，便于返回对应退出码。"""

    def __init__(self, signum: int):
        super().__init__(f"收到信号 {signum}")
        self.signum = signum


def run_child(main) -> int:
    directory = Path(os.environ[RUN_DIRECTORY_ENV])
    faulthandler.enable(all_threads=True)

    def interrupt(signum, _frame):
        raise TrainingInterrupted(signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        main()
    except BaseException as error:  # noqa: BLE001
        record_failure(directory, error)
        traceback.print_exc()
        if isinstance(error, KeyboardInterrupt):
            return 128 + getattr(error, "signum", signal.SIGINT)
        if isinstance(error, SystemExit):
            return error.code if isinstance(error.code, int) and error.code != 0 else 1
        return 1
    write_status(directory, state="completed")
    return 0


def supervise(script: Path) -> int:
    """在导入 torch、Warp 和 Isaac Lab 之前创建日志并启动训练子进程。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--run-name")
    parser.add_argument("--algorithm")
    parser.add_argument("--recurrent-type")
    parser.add_argument("--resume")
    args, _ = parser.parse_known_args()
    # 帮助信息无需创建空的运行目录。
    if "--help" in sys.argv or "-h" in sys.argv:
        environment = dict(os.environ, TA_SRU_LOGGED_RUN_DIRECTORY="")
        return subprocess.call(
            [sys.executable, "-u", str(script), *sys.argv[1:]], env=environment
        )
    prefix = "ppo" if args.algorithm == "ppo" else args.recurrent_type
    if prefix is None and args.resume:
        prefix = Path(args.resume).resolve().parent.parent.name.split("_", 1)[0]
    prefix = {"nn.lstm": "lstm"}.get(prefix, (prefix or "sru-lstm").replace("_", "-"))
    name = args.run_name or datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    directory = (Path(args.log_dir) / f"{prefix}_{name}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    write_status(
        directory,
        state="running",
        started_at=_now(),
        ended_at=None,
        exit_code=None,
        last_training_step=0,
        supervisor_pid=os.getpid(),
    )
    environment = dict(os.environ)
    environment.update(
        {
            RUN_DIRECTORY_ENV: str(directory),
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    print(f"训练日志：{directory}", flush=True)
    child = None
    previous_handlers = {}
    try:
        with (directory / "console.log").open("wb", buffering=0) as log:
            child = subprocess.Popen(
                [sys.executable, "-u", str(script), *sys.argv[1:]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            write_status(directory, child_pid=child.pid)

            def forward(signum, _frame):
                # 子进程独立成组，终端信号仅由监控进程转发一次。
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, forward)
            while chunk := child.stdout.read1(65536):
                log.write(chunk)
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except (BrokenPipeError, OSError):
                    # 终端消失时仍持续写入文件，避免丢失训练错误。
                    pass
            child.stdout.close()
            returncode = child.wait()
    except BaseException as error:  # noqa: BLE001
        record_failure(directory, error)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        returncode = 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    status = json.loads((directory / "status.json").read_text())
    exit_code = returncode if returncode >= 0 else 128 - returncode
    state = status["state"]
    if (returncode != 0 and state != "interrupted") or state == "running":
        state = "failed"
    if state != "completed" and exit_code == 0:
        exit_code = 1
    step = 0
    try:
        with (directory / "progress.csv").open() as progress:
            for row in csv.DictReader(progress):
                step = int(row["timesteps"])
    except (OSError, ValueError, KeyError):
        pass
    write_status(
        directory,
        state=state,
        ended_at=_now(),
        exit_code=exit_code,
        raw_returncode=returncode,
        signal=-returncode if returncode < 0 else None,
        last_training_step=max(step, status.get("last_training_step", 0)),
    )
    print(f"训练退出：state={state} exit_code={exit_code} 日志={directory}", flush=True)
    return exit_code or (0 if state == "completed" else 1)
