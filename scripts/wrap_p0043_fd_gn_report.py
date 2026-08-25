#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

p = Path("validation/reference_data/p0043_f_mapping_reidentification_v1/fd_gn_one_step.json")
r = json.loads(p.read_text())
trial = r["accepted_trial"]
r["final_eta"] = trial["eta"] if trial else []
r["final_parameters"] = trial["parameters"] if trial else {}
p.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
