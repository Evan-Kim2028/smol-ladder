"""Run a real transformers Trainer with `upload_folder` replaced by a copy into a directory.

    python trainer_hub_layout.py <dir> <hub_strategy>

Used by tests/test_hub_layout.py to capture which files the Hub repo would hold under each
hub_strategy. Needs torch, transformers 5.x and peft; no network.
"""
import sys, os, shutil, json, tempfile
from pathlib import Path
import torch, transformers
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from peft import LoraConfig, get_peft_model
import transformers.trainer as T
hub = Path(sys.argv[1]); strategy = sys.argv[2]
class Api:
    def create_repo(self, *a, **k):
        class R: repo_id='ns/x'
        return R()
    def whoami(self,*a,**k): return {'name':'ns'}
    def upload_folder(self, repo_id, folder_path, path_in_repo=None, ignore_patterns=None, run_as_future=False, **kw):
        import fnmatch
        dest = hub / (path_in_repo or "")
        for p in Path(folder_path).rglob("*"):
            rel = p.relative_to(folder_path)
            if p.is_file() and not any(fnmatch.fnmatch(rel.parts[0], pat) for pat in (ignore_patterns or [])):
                (dest/rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, dest/rel)
        class F:
            def done(self): return True
            def is_done(self): return True
            def result(self): return None
        return F()
T.hf_api = lambda: Api()
model = get_peft_model(GPT2LMHeadModel(GPT2Config(n_layer=1,n_head=2,n_embd=16,vocab_size=50)), LoraConfig(r=2, target_modules=["c_attn"], fan_in_fan_out=True))
class DS(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self,i): return {"input_ids": torch.arange(8)%50, "labels": torch.arange(8)%50}
out = Path(tempfile.mkdtemp())/"out"
args = TrainingArguments(output_dir=str(out), max_steps=4, save_steps=2, per_device_train_batch_size=2, push_to_hub=True, hub_model_id="ns/x", hub_strategy=strategy, report_to=[], use_cpu=True)
tr = Trainer(model=model, args=args, train_dataset=DS())
tr.create_model_card = lambda *a, **k: None
tr.init_hf_repo = lambda *a, **k: None
tr.train()
tr._finish_current_push() if hasattr(tr,"push_in_progress") else None
print(sorted(str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_file()))
print(json.loads((hub/"last-checkpoint/trainer_state.json").read_text())["global_step"] if (hub/"last-checkpoint").exists() else "NO last-checkpoint")
