import os

def _is_main_process():
    return os.getenv("RANK", "0") in ("0", "") or os.getenv("LOCAL_RANK", "0") in ("0", "")

def guarded_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)

# --- Helper: Push trained model and processor to Hugging Face Hub ---
def push_trained_model_to_hub(
        output_dir: str,
        repo_id: str,
        private: bool = True,
        use_auth_token: str | None = 'hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
):
    """
    Convenience helper to export a finished run to the Hugging Face Hub.

    Assumes that `train_ssl_mae` has already run and that:
      - The model was saved via `model.save_pretrained(output_dir)`.
      - The processor was saved under `os.path.join(output_dir, "checkpoints")`.

    Parameters
    ----------
    output_dir : str
        The same run folder that was passed as `output_dir` into `train_ssl_mae`.
        This folder must contain the saved model (config + weights).
    repo_id : str
        Target Hub repository in the form "username/repo_name".
    private : bool, optional
        If True, create/upload to a private repo. Defaults to True.
    use_auth_token : str or None, optional
        Optional Hugging Face token. If None, the token is taken from the
        environment or `huggingface-cli login` credentials.
    """
    from transformers import ViTMAEForPreTraining, ViTImageProcessor

    # Load trained model from the run root
    model = ViTMAEForPreTraining.from_pretrained(output_dir)

    # By construction in `train_ssl_mae`, the processor is stored under "checkpoints"
    processor_dir = os.path.join(output_dir, "checkpoints")
    processor = ViTImageProcessor.from_pretrained(processor_dir)

    # Build kwargs compatible with older / newer transformers versions
    push_kwargs = {}
    if use_auth_token is not None:
        push_kwargs["use_auth_token"] = use_auth_token
    # `private` is supported in recent transformers; if not, this will be ignored downstream
    push_kwargs["private"] = private

    # `push_to_hub` will create the repo if it does not exist and you are authenticated
    guarded_print(f"[HF Export] Pushing model to Hub repo='{repo_id}', private={private}")
    model.push_to_hub(repo_id, **push_kwargs)
    processor.push_to_hub(repo_id, **push_kwargs)
    guarded_print("[HF Export] Push complete.")

