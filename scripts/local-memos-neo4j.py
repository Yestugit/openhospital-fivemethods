"""Manage a project-local Neo4j Community for MemOS, without a Windows service.

Run with .venv-memos/Scripts/python.exe. `test` runs controlled native tests,
not clinical experiments. Passwords and all downloaded/runtime files stay under
the ignored .mematlas/services directory. Existing installations are preserved.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

VERSION = "5.26.0"
BOLT_PORT, HTTP_PORT = 17687, 17474
SHA256 = "ea9c98111b72310dab8ae00b7ae4d43df59a1ea82a490579b886bffb3e30de8f"
ROOT = Path(__file__).resolve().parents[1]
SERVICES = ROOT / ".mematlas" / "services"
HOME = SERVICES / ("neo4j-community-" + VERSION)
IDENTITY = SERVICES / "memos-neo4j-local.json"
URI = "bolt://127.0.0.1:17687"
LAUNCHER = ROOT / "scripts" / "LocalMemOSNeo4j.java"
STOP_FILE = HOME / "run" / "mematlas-stop-signal"


def java_environment():
    java = shutil.which("java")
    if not java:
        raise RuntimeError("Neo4j requires an installed Java 21; java is not on PATH")
    output = subprocess.run([java, "-XshowSettings:properties", "-version"],
                            capture_output=True, text=True, check=True)
    home = re.search(r"java.home\s*=\s*(.+)", output.stderr)
    if not home:
        raise RuntimeError("Cannot determine Java home")
    version = re.search(r"java.version\s*=\s*(\d+)", output.stderr)
    if not version or version.group(1) != "21":
        raise RuntimeError("This verified Neo4j deployment uses JDK 21; select Java 21 on PATH")
    env = dict(os.environ)
    env["JAVA_HOME"] = home.group(1).strip()
    env["NEO4J_HOME"] = str(HOME)
    return env


def identity():
    if not IDENTITY.is_file():
        raise RuntimeError("Run start first to initialize the local database")
    return json.loads(IDENTITY.read_text(encoding="utf-8"))


def install(env):
    SERVICES.mkdir(parents=True, exist_ok=True)
    archive = SERVICES / (HOME.name + "-windows.zip")
    if not archive.is_file():
        cached = ROOT / ".mematlas/services" / archive.name
        if cached.is_file() and cached.resolve() != archive.resolve():
            shutil.copyfile(cached, archive)
    if not archive.is_file():
        partial = archive.with_suffix(".download")
        with urlopen("https://dist.neo4j.org/" + archive.name, timeout=60) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target)
        partial.replace(archive)
    with archive.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != SHA256:
            raise RuntimeError("Neo4j download checksum mismatch; archive preserved for inspection")
    if not HOME.exists():
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                if not (SERVICES / item.filename).resolve().is_relative_to(SERVICES.resolve()):
                    raise RuntimeError("Unexpected archive path")
            source.extractall(SERVICES)
    if IDENTITY.exists():
        return identity()
    if (HOME / "data" / "databases" / "system").exists():
        raise RuntimeError("Existing database without a local identity; preserving its configuration")
    config = HOME / "conf" / "neo4j.conf"
    original = config.read_text(encoding="utf-8")
    backup = config.with_suffix(".conf.distribution")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    settings = {
        "server.default_listen_address": "127.0.0.1",
        "server.default_advertised_address": "127.0.0.1",
        "server.bolt.enabled": "true",
        "server.bolt.listen_address": f"127.0.0.1:{BOLT_PORT}",
        "server.bolt.advertised_address": f"127.0.0.1:{BOLT_PORT}",
        "server.http.enabled": "true",
        "server.http.listen_address": f"127.0.0.1:{HTTP_PORT}",
        "server.http.advertised_address": f"127.0.0.1:{HTTP_PORT}",
        "server.https.enabled": "false",
        "server.memory.heap.initial_size": "256m",
        "server.memory.heap.max_size": "512m",
        "server.memory.pagecache.size": "128m",
        "dbms.usage_report.enabled": "false",
    }
    lines = [line for line in original.splitlines()
             if line.partition("=")[0].strip() not in settings]
    config.write_text("\n".join(lines) + "\n\n# MemAtlas local MemOS database\n"
                      + "\n".join(key + "=" + value for key, value in settings.items()) + "\n", encoding="utf-8")
    data = {"version": VERSION, "uri": URI, "user": "neo4j", "database": "neo4j",
            "password": secrets.token_urlsafe(32), "archive_sha256": SHA256}
    result = subprocess.run([str(HOME / "bin" / "neo4j-admin.bat"), "dbms", "set-initial-password",
                             data["password"], "--require-password-change=false"],
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Neo4j initial password setup failed: "
                           + (result.stderr + result.stdout).replace(data["password"], "[redacted]"))
    with IDENTITY.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
    return data


def probe(data):
    from neo4j import GraphDatabase
    with GraphDatabase.driver(data["uri"], auth=(data["user"], data["password"]),
                              connection_timeout=2, connection_acquisition_timeout=2) as driver:
        info = driver.get_server_info()
        with driver.session(database=data["database"]) as session:
            session.run("RETURN 1 AS ready").consume()
        return info.agent


def start(env):
    data = install(env)
    try:
        print("Already ready:", probe(data), data["uri"], flush=True)
        return data
    except Exception:
        pass
    # Use Neo4j's own generated Java command, preserving the distribution's JVM
    # options. Parse Windows argv without evaluating shell text or installing a service.
    result = subprocess.run([str(HOME / "bin" / "neo4j.bat"), "console", "--dry-run"],
                            env=env, capture_output=True, text=True, check=True)
    command = next((line for line in result.stdout.splitlines() if "CommunityEntryPoint" in line), None)
    if command is None:
        raise RuntimeError("Neo4j did not provide its console Java command")
    count = ctypes.c_int()
    split = ctypes.windll.shell32.CommandLineToArgvW
    split.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    split.restype = ctypes.POINTER(ctypes.c_wchar_p)
    pointer = split(command, ctypes.byref(count))
    if not pointer:
        raise RuntimeError("Cannot parse Neo4j Java command")
    try:
        arguments = [pointer[i] for i in range(count.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.cast(pointer, ctypes.c_void_p))
    if not Path(arguments[0]).is_file() or "org.neo4j.server.CommunityEntryPoint" not in arguments:
        raise RuntimeError("Unexpected Neo4j startup command")
    arguments[arguments.index("org.neo4j.server.CommunityEntryPoint")] = str(LAUNCHER)
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.unlink(missing_ok=True)
    env["MEMATLAS_NEO4J_STOP_FILE"] = str(STOP_FILE)
    with (HOME / "logs" / "local-console.log").open("ab") as output:
        process = subprocess.Popen(arguments, env=env, cwd=HOME, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=output,
                                   creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    data["pid"] = process.pid
    IDENTITY.write_text(json.dumps(data, indent=2), encoding="utf-8")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Neo4j exited; see " + str(HOME / "logs" / "local-console.log"))
        try:
            print("Ready:", probe(data), data["uri"], flush=True)
            return data
        except Exception:
            time.sleep(1)
    raise RuntimeError("Neo4j startup timed out; inspect its logs before starting again")


def stop(data):
    # Guard against a recycled PID before stopping this dedicated Java process.
    pid = int(data.get("pid", 0))
    if not pid:
        raise RuntimeError("No managed Neo4j PID recorded")
    command = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        print("Already stopped")
        return
    current = json.loads(result.stdout)
    if (str(LAUNCHER) not in current.get("CommandLine", "")
            or str(HOME) not in current["CommandLine"]):
        raise RuntimeError("Recorded PID does not belong to this local Neo4j; leaving it untouched")
    STOP_FILE.write_text("stop\n", encoding="utf-8")
    # SYNCHRONIZE handle: wait for shutdown hooks to finish, without killing a
    # possibly busy database or enumerating unrelated Java processes.
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if handle:
        try:
            if kernel.WaitForSingleObject(handle, 45000) != 0:
                raise RuntimeError("Neo4j is still stopping; inspect its log before starting again")
        finally:
            kernel.CloseHandle(handle)
    print("Stopped local Neo4j; its database and credentials are preserved")


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services-dir", type=Path)
    parser.add_argument("--bolt-port", type=int, default=17687)
    parser.add_argument("--http-port", type=int, default=17474)
    parser.add_argument("action", choices=("start", "stop", "status", "test"))
    args = parser.parse_args()
    global SERVICES, HOME, IDENTITY, URI, STOP_FILE, BOLT_PORT, HTTP_PORT
    BOLT_PORT, HTTP_PORT = args.bolt_port, args.http_port
    if args.services_dir:
        SERVICES = args.services_dir.resolve()
        HOME = SERVICES / ("neo4j-community-" + VERSION)
        IDENTITY = SERVICES / "memos-neo4j-local.json"
        STOP_FILE = HOME / "run" / "mematlas-stop-signal"
    URI = f"bolt://127.0.0.1:{BOLT_PORT}"
    if os.name != "nt":
        raise SystemExit("This local deployment helper targets Windows; use an external Neo4j on other platforms")
    if args.action == "stop":
        stop(identity())
    elif args.action == "status":
        data = identity()
        print("Ready:", probe(data), data["uri"])
    else:
        env = java_environment()
        data = start(env)
        if args.action == "test":
            env.update({"MEMATLAS_TEST_NEO4J_URI": data["uri"],
                        "MEMATLAS_TEST_NEO4J_PASSWORD": data["password"],
                        "MEMATLAS_TEST_NEO4J_USER": data["user"],
                        "MEMATLAS_TEST_NEO4J_DATABASE": data["database"]})
            return subprocess.call([sys.executable, str(ROOT / "scripts" / "check-memory-adapters.py"),
                                    "--unittest", "test/memos/test_memos_native.py"], cwd=ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
