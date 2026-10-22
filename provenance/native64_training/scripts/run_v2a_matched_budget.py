"""Authorized-only fresh V2-A matched-budget trainer; imports do no GPU work."""
from tiny_avatar_diffusion.v2_matched_budget_runner import main_for_variant

if __name__ == "__main__":
    main_for_variant("v2_a")
