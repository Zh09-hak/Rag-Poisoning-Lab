import os
import sys
import time
import requests

BASE = os.getenv("LAB_BASE_URL", "http://127.0.0.1:8000")

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status} {name} {detail}")
    if not condition:
        sys.exit(1)

def main():
    debug = requests.get(f"{BASE}/debug", timeout=30).json()
    check("debug_endpoint", isinstance(debug, dict), str(debug)[:120])

    settings = requests.get(f"{BASE}/settings", timeout=10).json()
    check("settings_endpoint", "provider" in settings, settings.get("provider", ""))

    qa = requests.post(f"{BASE}/selftest", json={"include_llm": False}, timeout=60).json()
    check("selftest", qa.get("passed") == qa.get("total"), f"{qa.get('passed')}/{qa.get('total')}")

    created = requests.post(
        f"{BASE}/create_claim",
        json={"document": "Smoke test claim for isolated QA.", "metadata": {"source": "smoke_test"}},
        timeout=20
    ).json()
    claim_id = created.get("id")
    check("create_claim", bool(claim_id), claim_id or "")

    updated = requests.post(
        f"{BASE}/update_claim",
        json={"id": claim_id, "document": "Smoke test claim updated for isolated QA."},
        timeout=20
    ).json()
    check("update_claim", updated.get("status") == "updated", updated.get("status", ""))

    store = requests.get(f"{BASE}/check_db", timeout=20).json()
    check("store_contains_claim", claim_id in store.get("ids", []), str(store.get("total_claims")))

    deleted = requests.post(f"{BASE}/delete_claims", json={"ids": [claim_id]}, timeout=20).json()
    check("delete_claim", deleted.get("deleted") == 1, str(deleted.get("deleted")))

    restored = requests.post(f"{BASE}/restore_claims", json={"items": deleted.get("backup", [])}, timeout=20).json()
    check("restore_claim", restored.get("restored") == 1, str(restored.get("restored")))

    requests.post(f"{BASE}/delete_claims", json={"ids": [claim_id]}, timeout=20)

    original = requests.get(f"{BASE}/settings", timeout=10).json()
    requests.post(
        f"{BASE}/settings",
        json={"deterministic_eval": True, "cache_enabled": True, "eval_temperature": 0.0},
        timeout=10
    )

    payload = {"question": "What is this person's professional background?", "top_k": 3, "threshold": 0.5}
    first = requests.post(f"{BASE}/evaluate", json=payload, timeout=180).json()
    check("evaluate_first", "acceptance_rate" in first, str(first.get("error", ""))[:120])

    second = requests.post(f"{BASE}/evaluate", json=payload, timeout=180).json()
    check("evaluate_cached", second.get("cached") is True, str(second.get("cached")))
    check("evaluate_stable", first.get("acceptance_rate") == second.get("acceptance_rate"), f"{first.get('acceptance_rate')} vs {second.get('acceptance_rate')}")

    requests.post(
        f"{BASE}/settings",
        json={
            "deterministic_eval": original.get("deterministic_eval", True),
            "cache_enabled": original.get("cache_enabled", True),
            "eval_temperature": original.get("eval_temperature", 0.0)
        },
        timeout=10
    )

    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()