"""Upload a model's output folder to the owner's PRIVATE Hugging Face dataset repo. The token is read from the
environment variable HF_TOKEN and is never printed, logged or written to disk."""
from __future__ import annotations

import os


def upload_folder(local_dir, repo_id, path_in_repo, log=print, ignore=None):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("  upload skipped: HF_TOKEN is not set"); return dict(uploaded=False, reason="no HF_TOKEN")
    if not repo_id:
        log("  upload skipped: no --hf-repo given"); return dict(uploaded=False, reason="no repo")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if not getattr(info, "private", True):
        raise RuntimeError(f"{repo_id} exists and is PUBLIC; refusing to upload. Make it private first.")
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=local_dir, path_in_repo=path_in_repo,
                      ignore_patterns=ignore)
    # verify sizes
    local = {}
    for root, _, files in os.walk(local_dir):
        for f in files:
            p = os.path.join(root, f); rel = os.path.relpath(p, local_dir)
            if ignore and any(__import__("fnmatch").fnmatch(rel, g) for g in ignore):
                continue
            local[rel] = os.path.getsize(p)
    remote = {}
    for e in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=path_in_repo, recursive=True):
        if hasattr(e, "size") and e.size is not None:
            remote[os.path.relpath(e.path, path_in_repo) if path_in_repo else e.path] = e.size
    bad = [k for k, v in local.items() if remote.get(k) != v]
    log(f"  uploaded {len(local)} files to {repo_id}/{path_in_repo}; size mismatches: {len(bad)}")
    return dict(uploaded=True, files=len(local), mismatches=bad[:20])
