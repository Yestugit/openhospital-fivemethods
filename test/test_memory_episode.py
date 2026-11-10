from mematlas import MemoryEpisode


def test_memory_episode_generates_unique_id_and_preserves_it_when_serialized():
    first = MemoryEpisode(task="first")
    second = MemoryEpisode(task="second")

    assert first.episode_id
    assert second.episode_id
    assert first.episode_id != second.episode_id
    assert MemoryEpisode.from_dict(first.to_dict()).episode_id == first.episode_id


def test_memory_episode_accepts_explicit_id():
    assert MemoryEpisode("existing-id").episode_id == "existing-id"
