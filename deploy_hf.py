"""
One-command deploy to Hugging Face Spaces.

NOTE: Hugging Face now requires a PRO subscription for Gradio Spaces. For free hosting use Render (see README).

    python deploy_hf.py                          # Space name: pkm-urdu-assistant
    python deploy_hf.py --name my-space --private

Needs a Hugging Face access token with WRITE permission:
    huggingface.co -> Settings -> Access Tokens -> Create new token (type: Write)
Put it in .env as HF_TOKEN=hf_...   (or run `huggingface-cli login` once).

If GEMINI_API_KEY is in .env it is stored as a Space *secret* (encrypted) - the .env file itself is never uploaded.
Run the same command again after any change to update the live Space.
"""
import argparse
import os

from llm import load_env

IGNORE = [".env", "__pycache__/*", "**/__pycache__/*", "data/cache/*",
          "data/*.jsonl", "*.zip", ".git/*", ".venv/*", "venv/*"]


def main():
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pkm-urdu-assistant", help="Space name")
    p.add_argument("--private", action="store_true", help="only you can see the Space")
    args = p.parse_args()

    load_env()
    api = HfApi(token=os.environ.get("HF_TOKEN"))   # None -> uses `huggingface-cli login`
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.name}"

    print(f"Creating/updating Space {repo_id} ...")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        api.add_space_secret(repo_id, "GEMINI_API_KEY", gemini_key)
        print("GEMINI_API_KEY saved as a Space secret.")
    else:
        print("No GEMINI_API_KEY in .env - the Space will run without Gemini (add the secret later in Settings).")

    api.upload_folder(folder_path=os.path.dirname(os.path.abspath(__file__)), repo_id=repo_id,
                      repo_type="space", ignore_patterns=IGNORE,
                      commit_message="Deploy PKM Urdu Assistant")
    print(f"\nDone. Build logs and app: https://huggingface.co/spaces/{repo_id}")
    print("The first build takes about 5-10 minutes.")


if __name__ == "__main__":
    main()
