#!/usr/bin/env python3
# Copyright © 2026 Faux Fox, Inc.
"""Launch one process per rank for fMLX tensor-parallel inference.

Reads the same JSON hostfile as `mlx.launch`, starts the command on every host over
ssh (or locally), and sets the environment MLX's ring or JACCL backend reads. Uses only
the Python standard library.

Hostfile (the `mlx.distributed_config` output works as is):

    {
      "backend": "jaccl",
      "envs": ["MLX_METAL_FAST_SYNCH=1"],
      "hosts": [
        {"ssh": "studio-1", "ips": ["10.0.0.1"], "rdma": [null, ["rdma_en2", "rdma_en3"]]},
        {"ssh": "studio-2", "ips": [],           "rdma": [["rdma_en2", "rdma_en3"], null]}
      ]
    }

A bare list of hosts is accepted too. For JACCL, `rdma[j]` on host `i` names the RDMA
device that connects host `i` to host `j`: one name per cable, or a list of names when a
pair is joined by several cables (device `k` on one side must connect to device `k` on
the other). Rank 0 needs an IP every other host can reach; it runs the JACCL
coordinator on `--starting-port`. The ring backend needs IPs on every host.

Before a large model will fit, raise the GPU wired-memory limit on every Mac. This
script never runs sudo; run it yourself on each host (the value resets on reboot):

    sudo sysctl iogpu.wired_limit_mb=<megabytes>   # e.g. 480000 on a 512 GB Mac Studio

MLX_METAL_FAST_SYNCH=1 lowers collective latency, which matters for JACCL, but it needs
the Metal `fence` kernel. The current FauxFoxIO/mlx-swift build does not ship that kernel
in its metallib, so every rank fails with "Unable to load kernel fence_wait". Pass
--fast-synch (or list it in the hostfile's "envs") once mlx-swift includes it.

Examples:

    # two local processes over the ring backend (no RDMA needed)
    scripts/fmlx-launch.py -n 2 -- /path/to/fmlx-tp-generate --model /models/Qwen3-8B-4bit

    # two Mac Studios over Thunderbolt 5 RDMA
    scripts/fmlx-launch.py --hostfile studios.json --backend jaccl -- \\
        /path/to/fmlx-tp-generate --model /models/GLM --prompt "Hello"
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

WIRED_LIMIT_HINT = (
    "Raise the GPU wired limit on every host before loading a large model "
    "(not done by this script): sudo sysctl iogpu.wired_limit_mb=<megabytes>"
)


class Host:
    def __init__(self, ssh, ips, rdma):
        self.ssh = ssh
        self.ips = ips
        self.rdma = rdma

    @property
    def is_local(self):
        return self.ssh in LOCAL_HOSTS


def parse_hostfile(path):
    with open(path) as f:
        data = json.load(f)
    backend, envs = None, []
    if isinstance(data, dict):
        backend = data.get("backend") or None
        envs = data.get("envs", [])
        data = data["hosts"]
    if not isinstance(data, list) or not data:
        raise ValueError("the hostfile must list at least one host")
    hosts = [Host(h["ssh"], h.get("ips", []), h.get("rdma", [])) for h in data]
    return hosts, backend, envs


def hosts_from_list(names, repeat):
    hosts = []
    for name in names.split(","):
        if not name:
            raise ValueError("empty host name")
        hosts.extend(Host(name, [name], []) for _ in range(repeat))
    return hosts


def validate_jaccl(hosts, ring_topology):
    n = len(hosts)
    if not hosts[0].ips:
        return "rank 0 needs an IP that every host can reach (the JACCL coordinator)"
    for i, host in enumerate(hosts):
        if len(host.rdma) != n:
            return f"{host.ssh} lists {len(host.rdma)} RDMA entries; expected {n}"
        if host.rdma[i] is not None:
            return f"{host.ssh}: its own RDMA entry (position {i}) must be null"
        peers = ((i - 1) % n, (i + 1) % n) if ring_topology else range(n)
        for j in peers:
            if j != i and not host.rdma[j]:
                return f"no RDMA device is listed from {host.ssh} to {hosts[j].ssh}"
    return None


def backend_environment(hosts, backend, starting_port, verbose):
    """The variables and files every rank gets for `backend`."""
    env, files = [], {}
    if backend == "ring":
        if any(not h.ips for h in hosts):
            raise ValueError("the ring backend needs an IP for every host")
        port = starting_port
        ring = []
        for host in hosts:
            ring.append([f"{ip}:{port + k}" for k, ip in enumerate(host.ips)])
            port += len(host.ips)
        files["MLX_HOSTFILE"] = json.dumps(ring)
        if verbose:
            env.append("MLX_RING_VERBOSE=1")
    elif backend in ("jaccl", "jaccl-ring"):
        problem = validate_jaccl(hosts, backend == "jaccl-ring")
        if problem:
            raise ValueError(f"malformed JACCL hostfile: {problem}")
        env.append(f"MLX_JACCL_COORDINATOR={hosts[0].ips[0]}:{starting_port}")
        if backend == "jaccl-ring":
            env.append("MLX_JACCL_RING=1")
        files["MLX_IBV_DEVICES"] = json.dumps([h.rdma for h in hosts])
    else:
        raise ValueError(f"unsupported backend {backend!r}")
    env.append(f"FMLX_BACKEND={'jaccl' if backend.startswith('jaccl') else 'ring'}")
    return env, files


def rank_script(rank, cwd, env, files, command):
    """A bash script that sets up one rank and execs the command."""
    script = []
    if cwd:
        script.append(f"cd {shlex.quote(cwd)} || exit 1")
    for item in env:
        key, _, value = item.partition("=")
        if not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment variable {item!r}")
        script.append(f"export {key}={shlex.quote(value)}")
    for name, content in files.items():
        script.append(f"f=$(mktemp) && printf '%s' {shlex.quote(content)} > \"$f\"")
        script.append(f"export {name}=\"$f\"")
    script.append(f"export MLX_RANK={rank}")
    script.append("exec " + shlex.join(command))
    return "; ".join(script)


def forward(stream, rank, out, prefix):
    for line in iter(stream.readline, b""):
        text = line.decode(errors="replace")
        out.write(f"[{rank}] {text}" if prefix else text)
        out.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hostfile", help="mlx.launch JSON hostfile")
    source.add_argument("--hosts", help="comma-separated IPs (ring backend)")
    parser.add_argument(
        "-n", "--repeat-hosts", type=int, default=1,
        help="processes per host; with no hosts, run this many on localhost",
    )
    parser.add_argument("--backend", choices=["ring", "jaccl", "jaccl-ring"])
    parser.add_argument(
        "--starting-port", type=int, default=32323,
        help="first ring port, or the JACCL coordinator port on rank 0",
    )
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE for every rank")
    parser.add_argument(
        "--fast-synch", action="store_true",
        help="set MLX_METAL_FAST_SYNCH=1, the faster CPU/GPU sync for collectives (see above)",
    )
    parser.add_argument("--cwd", help="working directory on every host")
    parser.add_argument("--dry-run", action="store_true", help="print each rank's script and exit")
    parser.add_argument("--no-prefix", action="store_true", help="do not prefix output with the rank")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- command [args...]")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; put it after --")

    try:
        if args.hostfile:
            hosts, file_backend, file_envs = parse_hostfile(args.hostfile)
            hosts = [h for h in hosts for _ in range(args.repeat_hosts)]
        else:
            names = args.hosts or "127.0.0.1"
            hosts, file_backend, file_envs = hosts_from_list(names, args.repeat_hosts), None, []
        backend = args.backend or file_backend or "ring"
        env = list(file_envs)
        if args.fast_synch and not any(e.startswith("MLX_METAL_FAST_SYNCH=") for e in env):
            env.append("MLX_METAL_FAST_SYNCH=1")
        backend_env, files = backend_environment(hosts, backend, args.starting_port, args.verbose)
        env += backend_env + args.env
        scripts = [rank_script(r, args.cwd, env, files, command) for r in range(len(hosts))]
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    if args.verbose or args.dry_run:
        print(f"backend {backend}, {len(hosts)} ranks. {WIRED_LIMIT_HINT}", file=sys.stderr)
    if args.dry_run:
        for rank, (host, script) in enumerate(zip(hosts, scripts)):
            where = "local" if host.is_local else f"ssh {host.ssh}"
            print(f"# rank {rank} ({where})\n{script}")
        return 0

    processes = []
    for rank, (host, script) in enumerate(zip(hosts, scripts)):
        if host.is_local:
            argv = ["/bin/bash", "-c", script]
        else:
            # -tt ties the remote process to this ssh session, so it ends when we do
            argv = ["ssh", "-tt", "-o", "LogLevel=QUIET", host.ssh, script]
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        processes.append(process)
        for stream, out in ((process.stdout, sys.stdout), (process.stderr, sys.stderr)):
            threading.Thread(
                target=forward, args=(stream, rank, out, not args.no_prefix), daemon=True
            ).start()

    def stop_all(*_):
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    # one failed rank leaves the others blocked in a collective, so stop them all
    status = 0
    while any(p.poll() is None for p in processes):
        for rank, process in enumerate(processes):
            code = process.poll()
            if code not in (None, 0) and status == 0:
                status = code
                print(f"rank {rank} exited with {code}; stopping the other ranks", file=sys.stderr)
                stop_all()
        time.sleep(0.1)
    for rank, process in enumerate(processes):
        if process.returncode != 0 and status == 0:
            status = process.returncode
    time.sleep(0.2)  # let the output threads drain
    return status


if __name__ == "__main__":
    sys.exit(main())
