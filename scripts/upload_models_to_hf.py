from pathlib import Path

from huggingface_hub import HfApi, login


# Run this file from the ThinkOmni/Code directory.
ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "ThinkOmni-Stage1-SFA": ROOT / "model" / "model_stage1_SFA",
    "ThinkOmni-Stage2-SFA-AFA": ROOT / "model" / "model_stage2_SFA+AFA",
    "ThinkOmni-Stage3-SFA-AFA-MFR": ROOT / "model" / "model_stage3_SFA+AFA+MFR",
}


def main():
    login()
    api = HfApi()

    for repo_name, folder in MODELS.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"Model directory not found: {folder}")

        repo_id = f"beyond0814/{repo_name}"
        print(f"\nUploading {folder} -> {repo_id}")

        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
        )

        # Resumable upload for large model folders.
        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            repo_type="model",
            num_workers=4,
        )

        print(f"Finished: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

