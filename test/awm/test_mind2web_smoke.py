import json

from mind2web_smoke import choose_short_pair, score_action


def sample(identity, website, actions):
    return {
        "annotation_id": identity,
        "website": website,
        "confirmed_task": f"Task {identity}",
        "actions": actions,
        "action_reprs": [f"Action {index}" for index, _ in enumerate(actions)],
    }


def test_pair_selection_uses_shortest_shared_website_and_is_deterministic():
    train = [sample("long", "shop", [{}, {}, {}]), sample("short", "shop", [{}]),
             sample("other", "travel", [{}])]
    test = [sample("test-long", "shop", [{}, {}]), sample("test-short", "shop", [{}]),
            sample("none", "food", [{}])]

    assert choose_short_pair(train, test) == (train[1], test[1])


def test_native_step_score_is_computed_after_prediction_without_reference_leak():
    target = "CLICK [12]"
    result = score_action("```CLICK [12]```", target, [{"backend_node_id": "12"}])

    assert result == {"element_acc": 1.0, "action_f1": 1.0, "step_success": 1.0}
    assert "SECRET" not in json.dumps(result)
