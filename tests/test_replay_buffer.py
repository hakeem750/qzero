import numpy as np

from replay.buffer import ReplayBuffer


def test_replay_buffer_save_load_round_trip(tmp_path):
    buffer = ReplayBuffer(capacity=8)
    for i in range(5):
        obs = np.full((20, 9, 9), i / 10, dtype=np.float32)
        policy = np.zeros(140, dtype=np.float32)
        policy[i] = 1.0
        buffer.push(obs, policy, value=float(i - 2), weight=float(i + 1))

    path = tmp_path / "buffer.npz"
    buffer.save(path)

    loaded = ReplayBuffer.load(path, capacity=8)

    assert loaded.size == buffer.size
    assert loaded._ptr == buffer._ptr
    assert np.array_equal(loaded._obs[:loaded.size], buffer._obs[:buffer.size])
    assert np.array_equal(loaded._policy[:loaded.size], buffer._policy[:buffer.size])
    assert np.array_equal(loaded._value[:loaded.size], buffer._value[:buffer.size])
    assert np.array_equal(loaded._weights[:loaded.size], buffer._weights[:buffer.size])
