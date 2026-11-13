from datetime import datetime, timedelta, timezone
from memory_mesh.core.types import MemoryEpisode
from memory_mesh.core.decay import EbbinghausDecayModel


def test_ebbinghaus_decay_initial_retention():
    now = datetime.now(timezone.utc)
    episode = MemoryEpisode(content="User mentioned favorite programming language is Python", importance=0.8, created_at=now)
    model = EbbinghausDecayModel(default_half_life_hours=24.0)

    # Initial retention at t=0 should be 1.0
    r0 = model.compute_retention(episode, current_time=now)
    assert abs(r0 - 1.0) < 1e-4


def test_decay_over_time_and_reinforcement():
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    model = EbbinghausDecayModel(default_half_life_hours=24.0)

    ep_unaccessed = MemoryEpisode(content="Temp observation", importance=0.5, created_at=t0)
    ep_reinforced = MemoryEpisode(content="Important core guideline", importance=0.5, created_at=t0)

    # Simulate 5 accesses for reinforced episode
    for _ in range(5):
        ep_reinforced.mark_accessed(t0)

    # 48 hours later
    t1 = t0 + timedelta(hours=48)
    r_unaccessed = model.compute_retention(ep_unaccessed, current_time=t1)
    r_reinforced = model.compute_retention(ep_reinforced, current_time=t1)

    assert r_reinforced > r_unaccessed
    assert model.compute_retention(ep_unaccessed, t1) < 1.0
