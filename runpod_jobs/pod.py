"""RunPod helpers shared by the CPU and GPU runners (no heavy imports)."""

from __future__ import annotations

import os


def stop_this_pod(reason: str) -> None:
    """Stop (not delete) the pod: compute billing ends, /workspace with all results
    and caches is kept (only the disk bills, ~$0.01/h for 50 GB). Start it again
    to copy results, then terminate it."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        return
    if not os.path.ismount("/workspace"):
        # no volume disk / network volume: /workspace lives on the container disk,
        # which a stop ERASES -- keep the pod running rather than lose the results
        print(f"[stop] {reason}. NOT stopping the pod: /workspace is not a persistent volume, so "
              "stopping would erase the results. Copy them off now, then terminate the pod.",
              flush=True)
        return
    import subprocess

    print(f"[stop] {reason}: stopping pod {pod} so it stops billing compute", flush=True)
    for cmd in (["runpodctl", "pod", "stop", pod], ["runpodctl", "stop", "pod", pod]):
        try:
            if subprocess.run(cmd, timeout=120).returncode == 0:
                return
        except (OSError, subprocess.TimeoutExpired):
            continue
    print("[stop] could not stop the pod automatically -- stop it in the console NOW", flush=True)
