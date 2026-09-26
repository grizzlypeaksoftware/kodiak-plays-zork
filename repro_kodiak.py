"""Minimal reproduction of the Kodiak behaviors noted in the README (CPU, no game or Jericho needed)."""

from kodiak_s1.hub import Kodiak

kodiak = Kodiak.from_pretrained("cortex-agent-llc/kodiak-small-r1-preview", device="cpu")
state = ("Inside Building\nYou are inside a building, a well house for a large spring.\n"
         "There are some keys on the ground here.\nThere is tasty food here.\n"
         "There is a shiny brass lamp nearby.\nThere is an empty bottle here.\n\nInventory: You are carrying nothing.")
labels = ["west", "take keys", "take lantern", "take bottle", "take food", "turn lantern on", "take all",
          "light lantern with bottle", "push keys to ground", "push bottle to stream", "say fee"]
a = kodiak.decide(state, [
    {"type": "choice", "id": "action", "text": "Which command best makes progress in this adventure?", "labels": labels},
    {"type": "choice", "id": "danger", "text": "Is the player in danger?", "labels": ["yes", "no"]},
])
for q in ("action", "danger"):
    top = sorted(a[q]["probs"].items(), key=lambda kv: -kv[1])[:4]
    print(q, "->", a[q]["answer"], f"conf {a[q]['confidence']:.2f}, p_null {a[q]['p_null']:.2f}, top:",
          ", ".join(f"{k} {v:.2f}" for k, v in top))
