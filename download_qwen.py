from pathlib import Path

from huggingface_hub import snapshot_download


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-1.7B"
LOCAL_DIR = SCRIPT_DIR / "Qwen3-1.7B"


def main():
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print("Done")


if __name__ == "__main__":
    main()
