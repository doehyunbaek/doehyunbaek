from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "DevcToDockerResult",
    "devc_to_docker",
    "register_subcommand",
]


@dataclass
class BindMount:
    source: Path
    destination: Path


@dataclass
class CopySpec:
    destination: Path
    host_path: Path


@dataclass
class DevcToDockerResult:
    container_id: str
    final_image: str
    intermediate_image: str
    temp_dir: Path
    copied_mounts: list[CopySpec]


@dataclass
class DevcontainerOptions:
    remote_env: dict[str, str]
    remote_user: str | None
    workspace_folder: str | None
    entrypoint: Sequence[str] | None


class DevcToDockerError(RuntimeError):
    """Domain-specific error with a friendlier message."""


def _ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        raise DevcToDockerError("Docker CLI not found on PATH. Install Docker or adjust PATH before running this tool.")


def _run_docker(args: Sequence[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", *args]
    print("$ " + " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _inspect_container(container_id: str) -> dict:
    result = _run_docker(["inspect", container_id], capture_output=True)
    info = json.loads(result.stdout)
    if not info:
        raise DevcToDockerError(f"Container {container_id!r} not found")
    return info[0]


def _extract_bind_mounts(mounts: Iterable[dict]) -> list[BindMount]:
    bind_mounts: list[BindMount] = []
    for mount in mounts:
        if mount.get("Type") != "bind":
            continue
        source = Path(mount.get("Source", ""))
        destination = Path(mount.get("Destination", ""))
        if not source:
            continue
        bind_mounts.append(BindMount(source=source, destination=destination))
    return bind_mounts


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "-", name).strip("-.")
    return sanitized or "container"


def _read_devcontainer_file(devcontainer_dir: Path) -> dict:
    config_path = devcontainer_dir / "devcontainer.json"
    if not config_path.exists():
        raise DevcToDockerError(f"Could not find devcontainer.json inside {devcontainer_dir}")
    if not config_path.is_file():
        raise DevcToDockerError(f"Expected {config_path} to be a file")
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise DevcToDockerError(f"Failed to parse {config_path}: {exc}") from exc


def _env_list_to_dict(env_list: Iterable[str]) -> dict[str, str]:
    env_dict: dict[str, str] = {}
    for item in env_list:
        key, sep, value = item.partition("=")
        if not sep:
            continue
        env_dict[key] = value
    return env_dict


_TOKEN_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_template(value: str, *, container_env: Mapping[str, str], workspace_folder: str | None) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith("containerEnv:"):
            key = token.split(":", 1)[1]
            return container_env.get(key, "")
        if token in {"containerWorkspaceFolder", "workspaceFolder"}:
            return workspace_folder or ""
        return match.group(0)

    return _TOKEN_PATTERN.sub(replace, value)


def _parse_devcontainer_options(
    devcontainer_dir: Path,
    *,
    container_env: Mapping[str, str],
    default_workdir: str | None,
) -> DevcontainerOptions:
    raw = _read_devcontainer_file(devcontainer_dir)

    env_templates: dict[str, str] = {}
    remote_env_raw = raw.get("remoteEnv") or {}
    for key, value in remote_env_raw.items():
        env_templates[str(key)] = str(value)

    remote_user = raw.get("remoteUser")
    workspace_folder = raw.get("workspaceFolder") or default_workdir
    entrypoint: Sequence[str] | None = None

    run_args = raw.get("runArgs") or []
    if isinstance(run_args, list) and run_args:
        idx = 0
        while idx < len(run_args):
            arg = run_args[idx]
            if arg == "--entrypoint":
                idx += 1
                if idx >= len(run_args):
                    raise DevcToDockerError("runArgs specifies --entrypoint but no value was provided")
                entrypoint = shlex.split(run_args[idx])
            elif arg.startswith("--entrypoint="):
                entrypoint = shlex.split(arg.split("=", 1)[1])
            elif arg in ("-e", "--env"):
                idx += 1
                if idx >= len(run_args):
                    raise DevcToDockerError("runArgs specifies an environment flag without KEY=VALUE")
                key_value = run_args[idx]
                key, sep, value = key_value.partition("=")
                if not sep:
                    raise DevcToDockerError(f"Invalid environment assignment in runArgs: {key_value!r}")
                env_templates[key] = value
            elif arg.startswith("-e") and arg not in {"-e"}:
                key_value = arg[2:]
                key, sep, value = key_value.partition("=")
                if sep:
                    env_templates[key] = value
            elif arg.startswith("--env="):
                key_value = arg.split("=", 1)[1]
                key, sep, value = key_value.partition("=")
                if sep:
                    env_templates[key] = value
            elif arg in ("-u", "--user"):
                idx += 1
                if idx >= len(run_args):
                    raise DevcToDockerError("runArgs specifies a user flag without value")
                remote_user = run_args[idx]
            elif arg.startswith("-u") and arg not in {"-u"}:
                remote_user = arg[2:]
            elif arg.startswith("--user="):
                remote_user = arg.split("=", 1)[1]
            elif arg in ("-w", "--workdir"):
                idx += 1
                if idx >= len(run_args):
                    raise DevcToDockerError("runArgs specifies a workdir flag without value")
                workspace_folder = run_args[idx]
            elif arg.startswith("-w") and arg not in {"-w"}:
                workspace_folder = arg[2:]
            elif arg.startswith("--workdir="):
                workspace_folder = arg.split("=", 1)[1]
            idx += 1

    remote_env: dict[str, str] = {}
    for key, template in env_templates.items():
        resolved = _resolve_template(template, container_env=container_env, workspace_folder=workspace_folder)
        remote_env[key] = resolved

    return DevcontainerOptions(
        remote_env=remote_env,
        remote_user=remote_user,
        workspace_folder=workspace_folder,
        entrypoint=entrypoint,
    )


def _resolve_temp_dir(temp_root: str | os.PathLike[str] | None) -> Path:
    root = Path(temp_root) if temp_root else Path(tempfile.gettempdir())
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="devc_to_docker_", dir=str(root)))


def _copy_bind_mounts_to_tmp(bind_mounts: Sequence[BindMount], temp_dir: Path) -> list[CopySpec]:
    copies: list[CopySpec] = []
    for mount in bind_mounts:
        if not mount.source.exists():
            raise DevcToDockerError(f"Bind mount source does not exist on host: {mount.source}")
        destination = mount.destination
        if destination.is_absolute():
            parts = destination.parts[1:]
            if not parts:
                raise DevcToDockerError("Cannot mirror bind mount with destination '/' into temporary directory")
            relative = Path(*parts)
        else:
            relative = destination

        host_target = temp_dir / relative
        host_target.parent.mkdir(parents=True, exist_ok=True)
        if host_target.exists():
            if host_target.is_dir():
                shutil.rmtree(host_target)
            else:
                host_target.unlink()

        if mount.source.is_dir():
            shutil.copytree(mount.source, host_target, symlinks=True)
        else:
            shutil.copy2(mount.source, host_target)

        copies.append(CopySpec(destination=destination, host_path=host_target))
        print(f"Copied bind mount {mount.source} → {host_target}")
    return copies


def _start_keepalive_container(image: str, container_name: str) -> None:
    keepalive = "while true; do sleep 3600; done"
    _run_docker(
        [
            "run",
            "-d",
            "--name",
            container_name,
            "--entrypoint",
            "/bin/sh",
            image,
            "-lc",
            keepalive,
        ]
    )


def _restore_bind_mounts(temp_container: str, copies: Sequence[CopySpec]) -> None:
    for spec in copies:
        destination = spec.destination
        host_path = spec.host_path
        destination_parent = destination.parent if destination.parent != destination else Path("/")
        destination_str = destination.as_posix() or "."
        parent_str = destination_parent.as_posix() or "."

        # Remove any existing content so the copy is authoritative.
        try:
            _run_docker(["exec", temp_container, "rm", "-rf", destination_str])
        except subprocess.CalledProcessError as exc:
            print(f"Warning: could not remove existing {destination_str} in {temp_container}: {exc}")

        try:
            _run_docker(["exec", temp_container, "mkdir", "-p", parent_str])
        except subprocess.CalledProcessError as exc:
            raise DevcToDockerError(f"Unable to create parent directory {parent_str} inside {temp_container}: {exc}")

        # Copy the snapshot back into the container.
        _run_docker(["cp", str(host_path), f"{temp_container}:{parent_str}"])
        print(f"Restored {host_path} → {temp_container}:{destination_str}")


def _build_commit_changes(config: dict, dev_options: DevcontainerOptions | None = None) -> list[str]:
    changes: list[str] = []
    entrypoint = config.get("Entrypoint")
    if dev_options and dev_options.entrypoint is not None:
        entrypoint = list(dev_options.entrypoint)
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    # TODO: Replace hardcoded entrypoint with value derived from devcontainer configuration.
    entrypoint = ["/usr/bin/bash"]

    cmd = config.get("Cmd")

    if entrypoint:
        changes.append(f"ENTRYPOINT {json.dumps(entrypoint)}")
    else:
        changes.append("ENTRYPOINT []")

    if cmd:
        changes.append(f"CMD {json.dumps(cmd)}")
    else:
        changes.append("CMD []")

    work_dir = config.get("WorkingDir")
    if dev_options and dev_options.workspace_folder:
        work_dir = dev_options.workspace_folder
    # TODO: Replace hardcoded workdir with workspaceFolder resolved from devcontainer.
    work_dir = "/workspaces/wasm-r3"
    if work_dir:
        changes.append(f"WORKDIR {work_dir}")

    user = config.get("User")
    if dev_options and dev_options.remote_user:
        user = dev_options.remote_user
    # TODO: Replace hardcoded user with remoteUser parsed from devcontainer.
    user = "vscode"
    if user:
        changes.append(f"USER {user}")

    if dev_options and dev_options.remote_env:
        for key, value in dev_options.remote_env.items():
            changes.append(f"ENV {key}={json.dumps(value)}")

    return changes


def devc_to_docker(
    container_id: str,
    devcontainer_dir: str | os.PathLike[str] | None = None,
    *,
    output_image: str | None = None,
    intermediate_image: str | None = None,
    temp_root: str | os.PathLike[str] | None = None,
    keep_temp: bool = False,
    keep_intermediate: bool = False,
) -> DevcToDockerResult:
    """Capture the current container and its bind-mounted workspace into a reusable image."""
    # Dynamic defaults (container name, timestamp) can't be expressed as parameter defaults
    # because Python evaluates default args at import time. Compute them at runtime instead.

    _ensure_docker_available()
    inspect_info = _inspect_container(container_id)

    container_name = inspect_info.get("Name", "").lstrip("/") or container_id
    sanitized_name = _sanitize_name(container_name)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    if output_image is None:
        output_image = f"devc_to_docker/{sanitized_name}:{timestamp}"

    if intermediate_image is None:
        intermediate_image = f"{output_image}-stage"

    config = inspect_info.get("Config", {})
    container_env_map = _env_list_to_dict(config.get("Env", []))

    dev_options: DevcontainerOptions | None = None
    if devcontainer_dir is not None:
        dev_dir_path = Path(devcontainer_dir).expanduser().resolve()
        if not dev_dir_path.exists():
            raise DevcToDockerError(f"Devcontainer directory {dev_dir_path} does not exist")
        if not dev_dir_path.is_dir():
            raise DevcToDockerError(f"Expected {dev_dir_path} to be a directory")
        dev_options = _parse_devcontainer_options(
            dev_dir_path,
            container_env=container_env_map,
            default_workdir=config.get("WorkingDir"),
        )
        print(f"Parsed devcontainer overrides from {dev_dir_path}")

    bind_mounts = _extract_bind_mounts(inspect_info.get("Mounts", []))
    if not bind_mounts:
        raise DevcToDockerError(f"Container {container_id} has no bind mounts – nothing to copy.")

    print(f"Using intermediate image tag: {intermediate_image}")
    print(f"Final image will be stored as: {output_image}")

    commit_stage = _run_docker(["commit", container_id, intermediate_image], capture_output=True)
    stage_image_id = commit_stage.stdout.strip()
    print(f"Committed running container to intermediate image {stage_image_id}")

    temp_dir = _resolve_temp_dir(temp_root)
    print(f"Temporary snapshot directory: {temp_dir}")
    copies = _copy_bind_mounts_to_tmp(bind_mounts, temp_dir)

    temp_container_name = f"devc-to-docker-{sanitized_name}-{timestamp}"
    if len(temp_container_name) > 60:
        temp_container_name = temp_container_name[:60]

    _start_keepalive_container(intermediate_image, temp_container_name)

    try:
        _restore_bind_mounts(temp_container_name, copies)

        changes = _build_commit_changes(config, dev_options)
        commit_cmd = ["commit"]
        for change in changes:
            commit_cmd.extend(["--change", change])
        commit_cmd.extend([temp_container_name, output_image])
        commit_final = _run_docker(commit_cmd, capture_output=True)
        final_image_id = commit_final.stdout.strip()
        print(f"Final image committed as {output_image} ({final_image_id})")
    finally:
        try:
            _run_docker(["stop", temp_container_name])
        except subprocess.CalledProcessError:
            pass
        try:
            _run_docker(["rm", temp_container_name])
        except subprocess.CalledProcessError:
            pass
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Temporary snapshot directory {temp_dir} removed")
        else:
            print(f"Temporary snapshot retained at {temp_dir}")

    if not keep_intermediate:
        try:
            _run_docker(["rmi", intermediate_image])
        except subprocess.CalledProcessError as exc:
            print(f"Warning: could not remove intermediate image {intermediate_image}: {exc}")

    return DevcToDockerResult(
        container_id=container_id,
        final_image=output_image,
        intermediate_image=intermediate_image,
        temp_dir=temp_dir,
        copied_mounts=list(copies),
    )


def _handle_command(args: argparse.Namespace) -> int:
    try:
        result = devc_to_docker(
            args.container_id,
            args.devcontainer_dir,
            output_image=args.output_image,
            intermediate_image=args.intermediate_image,
            temp_root=args.temp_root,
            keep_temp=args.keep_temp,
            keep_intermediate=args.keep_intermediate,
        )
    except (DevcToDockerError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\nSummary:")
    print(f"  Container captured: {result.container_id}")
    print(f"  Final image tag:   {result.final_image}")
    print(f"  Intermediate tag: {result.intermediate_image}")
    if result.copied_mounts:
        print("  Copied bind mounts:")
        for spec in result.copied_mounts:
            print(f"    {spec.host_path} ← {spec.destination}")
    print(f"  Temporary dir:     {result.temp_dir}")
    return 0


def register_subcommand(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "devc_to_docker",
        help="Capture a VS Code dev container (including bind mounts) into a reusable Docker image.",
        description="Commit a running devcontainer and fold its bind-mounted workspace into the resulting image.",
    )
    parser.add_argument("container_id", help="ID or name of the running VS Code container")
    parser.add_argument("devcontainer_dir", help="Path to the .devcontainer directory to mirror configuration from")
    parser.add_argument(
        "-o",
        "--output-image",
        dest="output_image",
        help="Target image tag for the final snapshot (default: devc_to_docker/<container>:<timestamp>)",
    )
    parser.add_argument(
        "--intermediate-image",
        dest="intermediate_image",
        help="Optional tag for the intermediate commit (default: <output>-stage)",
    )
    parser.add_argument(
        "--temp-root",
        dest="temp_root",
        help="Directory under which temporary workspace archives should be created (default: system /tmp)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Retain the temporary snapshot directory for inspection",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the intermediate Docker image instead of deleting it",
    )
    parser.set_defaults(_handler=_handle_command)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devc_to_docker")
    parser.add_argument("container_id")
    parser.add_argument("devcontainer_dir")
    parser.add_argument("--output-image")
    parser.add_argument("--intermediate-image")
    parser.add_argument("--temp-root")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args(argv)
    return _handle_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
