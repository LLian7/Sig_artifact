from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
WORKER_PATH = ROOT / "benchmark_true_cycles_worker.py"
XCTRACE_TEMPLATE_NAME = "CPU Counters"
XCTRACE_PROCESS_TABLE_XPATH = (
    '/trace-toc/run[@number="1"]/data/table[@schema="MetricAggregationForProcess"]'
)
XCTRACE_PROCESS_DISCOVERY_DELAY_SECONDS = 2.0
XCTRACE_ATTACH_READY_TIMEOUT_SECONDS = 30.0
XCTRACE_ATTACH_SETTLE_SECONDS = 2.0
XCTRACE_FINALIZE_TIMEOUT_SECONDS = float(os.environ.get("XCTRACE_FINALIZE_TIMEOUT_SECONDS", "600"))
XCTRACE_INTERRUPT_GRACE_SECONDS = 15.0
RETRYABLE_XCTRACE_FAILURES = (
    "could not lock kperf",
    "Could not set the recording priority",
    "xctrace finalize timed out",
    "Document Missing Template Error",
    "Unexpected internal error",
)


class _XctraceRecordFailure(RuntimeError):
    def __init__(self, command: Sequence[str], stdout: str, stderr: str) -> None:
        self.command = command
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            "xctrace record failed.\n"
            f"command: {command!r}\n"
            f"xctrace stdout: {stdout}\n"
            f"xctrace stderr: {stderr}"
        )

    @property
    def retryable(self) -> bool:
        combined_output = f"{self.stdout}\n{self.stderr}"
        return any(message in combined_output for message in RETRYABLE_XCTRACE_FAILURES)


def _resolve_reference_value(element: ET.Element, id_map: Dict[str, object]) -> object:
    ref = element.attrib.get("ref")
    if ref is not None:
        return id_map[ref]

    tag = element.tag
    if tag in {"pid", "tid", "uint64"}:
        value: object = int(element.text or "0")
    elif tag == "boolean":
        text = (element.text or "").strip()
        if text:
            value = text not in {"0", "false", "False"}
        else:
            value = element.attrib.get("fmt") == "Yes"
    elif tag == "string":
        value = element.text if element.text is not None else element.attrib.get("fmt", "")
    elif tag == "process":
        pid_element = element.find("pid")
        value = None if pid_element is None else _resolve_reference_value(pid_element, id_map)
    else:
        value = element.text if element.text is not None else element.attrib.get("fmt")

    element_id = element.attrib.get("id")
    if element_id is not None:
        id_map[element_id] = value
    return value


def parse_xctrace_process_cycles(
    xml_text: str,
    *,
    expected_pid: int | None = None,
    precise_only: bool = True,
) -> int:
    root = ET.fromstring(xml_text)
    id_map: Dict[str, object] = {}
    total_cycles = 0
    found_process_table = False

    for node in root.iter("node"):
        schema = node.find("schema")
        if schema is None or schema.attrib.get("name") != "MetricAggregationForProcess":
            continue
        found_process_table = True
        for row in node.findall("row"):
            row_pid: int | None = None
            metric_value_int: int | None = None
            metric_name: str | None = None
            is_precise = False

            for child in row:
                value = _resolve_reference_value(child, id_map)
                if child.tag == "process":
                    row_pid = value if isinstance(value, int) else None
                elif child.tag == "uint64":
                    metric_value_int = value if isinstance(value, int) else None
                elif child.tag == "string":
                    metric_name = str(value)
                elif child.tag == "boolean":
                    is_precise = bool(value)

            if metric_name != "cycle" or metric_value_int is None:
                continue
            if precise_only and not is_precise:
                continue
            if expected_pid is not None and row_pid != expected_pid:
                continue
            total_cycles += metric_value_int

    if not found_process_table:
        raise ValueError("xctrace export did not contain MetricAggregationForProcess")
    return total_cycles


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5.0)


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        _send_process_group_signal(proc, signal.SIGTERM)
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            _send_process_group_signal(proc, signal.SIGKILL)
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    except ProcessLookupError:
        return


def _send_process_group_signal(proc: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except ProcessLookupError:
        return
    except PermissionError:
        proc.send_signal(sig)


def _finalize_xctrace_recording(
    trace_proc: subprocess.Popen[str],
    trace_stdout_prefix: str,
) -> tuple[str, str]:
    if trace_proc.poll() is None:
        _send_process_group_signal(trace_proc, signal.SIGINT)
        try:
            trace_stdout_tail, trace_stderr = trace_proc.communicate(
                timeout=XCTRACE_INTERRUPT_GRACE_SECONDS,
            )
            return trace_stdout_prefix + trace_stdout_tail, trace_stderr
        except subprocess.TimeoutExpired:
            _send_process_group_signal(trace_proc, signal.SIGINT)

    try:
        trace_stdout_tail, trace_stderr = trace_proc.communicate(
            timeout=XCTRACE_FINALIZE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        partial_stdout = exc.stdout or ""
        partial_stderr = exc.stderr or ""
        if isinstance(partial_stdout, bytes):
            partial_stdout = partial_stdout.decode(errors="replace")
        if isinstance(partial_stderr, bytes):
            partial_stderr = partial_stderr.decode(errors="replace")
        _terminate_process_group(trace_proc)
        try:
            trace_stdout_tail, trace_stderr = trace_proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            trace_stdout_tail, trace_stderr = "", ""
        trace_stdout = (
            trace_stdout_prefix
            + partial_stdout
            + trace_stdout_tail
            + "\n[xctrace finalize timed out; discarding incomplete trace]\n"
        )
        raise _XctraceRecordFailure(
            trace_proc.args,
            trace_stdout,
            partial_stderr + trace_stderr,
        )
    return trace_stdout_prefix + trace_stdout_tail, trace_stderr


def _read_worker_ready_pid(worker_proc: subprocess.Popen[str], timeout_seconds: float) -> int:
    if worker_proc.stdout is None:
        raise RuntimeError("worker stdout pipe was not created")
    deadline = time.monotonic() + timeout_seconds
    seen_stdout: list[str] = []

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(worker_proc.args, timeout_seconds)

        if worker_proc.poll() is not None:
            stdout_tail = worker_proc.stdout.read() if worker_proc.stdout is not None else ""
            stderr_tail = worker_proc.stderr.read() if worker_proc.stderr is not None else ""
            raise RuntimeError(
                "cycle worker exited before reporting readiness.\n"
                f"command: {worker_proc.args!r}\n"
                f"stdout: {''.join(seen_stdout)}{stdout_tail}\n"
                f"stderr: {stderr_tail}"
            )

        readable, _, _ = select.select([worker_proc.stdout], [], [], min(0.25, remaining))
        if not readable:
            continue
        line = worker_proc.stdout.readline()
        if not line:
            continue
        seen_stdout.append(line)
        if line.startswith("READY "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"malformed worker readiness line: {line!r}") from exc


def _wait_for_xctrace_attach_ready(
    trace_proc: subprocess.Popen[str],
    timeout_seconds: float,
) -> str:
    if trace_proc.stdout is None:
        raise RuntimeError("xctrace stdout pipe was not created")
    deadline = time.monotonic() + timeout_seconds
    seen_stdout: list[str] = []

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(trace_proc.args, timeout_seconds)

        if trace_proc.poll() is not None:
            stdout_tail = trace_proc.stdout.read() if trace_proc.stdout is not None else ""
            stderr_tail = trace_proc.stderr.read() if trace_proc.stderr is not None else ""
            raise _XctraceRecordFailure(
                trace_proc.args,
                "".join(seen_stdout) + stdout_tail,
                stderr_tail,
            )

        readable, _, _ = select.select([trace_proc.stdout], [], [], min(0.25, remaining))
        if not readable:
            continue
        line = trace_proc.stdout.readline()
        if not line:
            continue
        seen_stdout.append(line)
        if "Attaching to:" in line or "Starting recording" in line:
            return "".join(seen_stdout)


def _record_attached_worker_cycles_once(
    worker_command: Sequence[str],
    trace_path: Path,
    *,
    cwd: str | None,
    env: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, str, str]:
    sync_worker_command = [*worker_command, "--sync-stdin"]
    worker_proc = subprocess.Popen(
        sync_worker_command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    trace_proc: subprocess.Popen[str] | None = None
    try:
        ready_pid = _read_worker_ready_pid(worker_proc, timeout_seconds)
        worker_pid = worker_proc.pid
        if ready_pid != worker_pid:
            raise RuntimeError(
                "cycle worker reported a different pid than subprocess.Popen.\n"
                f"reported pid: {ready_pid}\n"
                f"popen pid: {worker_pid}"
            )

        # xctrace can miss a freshly spawned process if attach starts immediately.
        time.sleep(XCTRACE_PROCESS_DISCOVERY_DELAY_SECONDS)
        trace_command = [
            "xctrace",
            "record",
            "--template",
            XCTRACE_TEMPLATE_NAME,
            "--output",
            str(trace_path),
            "--attach",
            str(worker_pid),
        ]
        trace_proc = subprocess.Popen(
            trace_command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        trace_stdout_prefix = _wait_for_xctrace_attach_ready(
            trace_proc,
            XCTRACE_ATTACH_READY_TIMEOUT_SECONDS,
        )
        time.sleep(XCTRACE_ATTACH_SETTLE_SECONDS)

        if worker_proc.stdin is None:
            raise RuntimeError("worker stdin pipe was not created")
        worker_proc.stdin.write("RUN\n")
        worker_proc.stdin.flush()

        worker_stdout, worker_stderr = worker_proc.communicate(timeout=timeout_seconds)
        trace_stdout, trace_stderr = _finalize_xctrace_recording(
            trace_proc,
            trace_stdout_prefix,
        )

        if worker_proc.returncode != 0:
            raise RuntimeError(
                "cycle worker failed while running the measured loop.\n"
                f"command: {sync_worker_command!r}\n"
                f"stdout: {worker_stdout}\n"
                f"stderr: {worker_stderr}"
            )
        if trace_proc.returncode != 0 and not trace_path.exists():
            raise _XctraceRecordFailure(trace_command, trace_stdout, trace_stderr)
        return worker_pid, trace_stdout, trace_stderr
    except Exception:
        if trace_proc is not None:
            _terminate_process(trace_proc)
        _terminate_process(worker_proc)
        raise


def measure_worker_cycles_with_xctrace(
    worker_args: Sequence[str],
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float = 3600.0,
) -> int:
    if sys.platform != "darwin":
        raise RuntimeError("xctrace true-cycle backend is only supported on macOS")

    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")

    worker_command = [sys.executable, str(WORKER_PATH), *worker_args]
    with tempfile.TemporaryDirectory(prefix="xctrace-cycles-") as temp_dir:
        last_record_failure: _XctraceRecordFailure | None = None
        last_export_failure: RuntimeError | None = None
        for attempt in range(3):
            trace_path = Path(temp_dir) / f"measurement-{attempt}.trace"
            try:
                worker_pid, trace_stdout, trace_stderr = _record_attached_worker_cycles_once(
                    worker_command,
                    trace_path,
                    cwd=cwd,
                    env=merged_env,
                    timeout_seconds=timeout_seconds,
                )
                try:
                    export = subprocess.run(
                        [
                            "xctrace",
                            "export",
                            "--input",
                            str(trace_path),
                            "--xpath",
                            XCTRACE_PROCESS_TABLE_XPATH,
                        ],
                        cwd=cwd,
                        env=merged_env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    message = (
                        "xctrace export failed after recording.\n"
                        f"trace path: {trace_path}\n"
                        f"record stdout: {trace_stdout}\n"
                        f"record stderr: {trace_stderr}\n"
                        f"export stdout: {exc.stdout}\n"
                        f"export stderr: {exc.stderr}"
                    )
                    export_failure = RuntimeError(message)
                    combined_output = f"{exc.stdout}\n{exc.stderr}"
                    if not any(
                        retryable in combined_output
                        for retryable in RETRYABLE_XCTRACE_FAILURES
                    ):
                        raise export_failure from exc
                    last_export_failure = export_failure
                    if attempt < 2:
                        time.sleep(2.0)
                        continue
                    raise export_failure from exc

                total_cycles = parse_xctrace_process_cycles(
                    export.stdout,
                    expected_pid=worker_pid,
                )
                if total_cycles <= 0:
                    message = (
                        "xctrace did not report a positive cycle count.\n"
                        f"worker command: {worker_command!r}\n"
                        f"xctrace stdout: {trace_stdout}\n"
                        f"xctrace stderr: {trace_stderr}"
                    )
                    measure_failure = RuntimeError(message)
                    combined_output = f"{trace_stdout}\n{trace_stderr}"
                    if any(
                        retryable in combined_output
                        for retryable in RETRYABLE_XCTRACE_FAILURES
                    ):
                        last_export_failure = measure_failure
                        if attempt < 2:
                            time.sleep(2.0)
                            continue
                    raise measure_failure
                return total_cycles
            except _XctraceRecordFailure as exc:
                last_record_failure = exc
                if not exc.retryable:
                    raise
            if attempt < 2:
                time.sleep(1.0)
        else:
            if last_export_failure is not None:
                raise last_export_failure
            if last_record_failure is not None:
                raise last_record_failure
            raise RuntimeError("xctrace record failed before producing a trace")
    raise RuntimeError("unreachable xctrace measurement state")
