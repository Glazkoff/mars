import json
import secrets
import sys
names = sys.argv[1:] or ["Annotator 1", "Annotator 2"]
cfg = {"port": 8787, "items": "human_sample_blind.jsonl", "db": "labels.db", "admin_token": secrets.token_urlsafe(24), "annotators": {secrets.token_urlsafe(18): n for n in names}}
json.dump(cfg, open("config.json", "w"), indent=1)
for t, n in cfg["annotators"].items():
    print(f"{n}: https://label.example.org/a/{t}/")
print(f"admin: https://label.example.org/admin/{cfg['admin_token']}/")
