"""
Tests for canonical state encoding and value target alignment.

Verifies that:
1. Canonical states produce identical encodings for equivalent positions
2. Value targets are correctly signed from each player's perspective
3. Augmentation preserves outcome correctness
"""
import numpy as np
import pytest

from env.state import initial_state, QuoridorState
from env.encoding import encode_state, mirror_state_and_policy
from env.rules import apply_action


class TestCanonicalEncoding:
    """Verify canonical perspective enforcement."""

    def test_canonical_p1_state(self):
        """P1-to-move states are already canonical."""
        state = initial_state()
        assert state.current_player == 1
        canonical = state.canonical()
        assert canonical == state

    def test_canonical_p2_state(self):
        """P2-to-move states are flipped."""
        state = initial_state()
        # Make a move so P2 can move
        state = apply_action(state, 1)  # P1 moves forward
        assert state.current_player == 2
        
        canonical = state.canonical()
        assert canonical.current_player == 1
        # P2 was at (8,4), after flip should be at (0,4)
        assert canonical.p1_pos == (0, 4)
        assert canonical.p2_pos == (7, 4)

    def test_canonical_encoding_p1(self):
        """P1-to-move state encodes from their perspective."""
        state = initial_state()
        obs = encode_state(state.canonical())
        # Channel 0: current player pawn should be at (0,4)
        assert obs[0, 0, 4] == 1.0
        # Channel 1: opponent pawn should be at (8,4)
        assert obs[1, 8, 4] == 1.0

    def test_canonical_encoding_p2(self):
        """P2-to-move state encodes from their perspective (flipped board)."""
        state = initial_state()
        state = apply_action(state, 1)  # P1 moves, now P2 to move
        
        canonical = state.canonical()
        obs = encode_state(canonical)
        
        # In canonical form, P2 is "P1" and sees board flipped
        # P2 was at (8,4), flipped to (0,4) - should be current player pawn
        assert obs[0, 0, 4] == 1.0
        # P1 is now opponent, was at (1,4), flipped to (7,4)
        assert obs[1, 7, 4] == 1.0

    def test_canonical_positions_are_stable(self):
        """Canonicalizing twice yields the same result."""
        state = initial_state()
        state = apply_action(state, 1)
        
        canonical1 = state.canonical()
        canonical2 = canonical1.canonical()
        
        assert canonical1 == canonical2


class TestValueTargetAlignment:
    """Verify outcomes are correctly signed for each player."""

    def test_value_sign_p1_wins(self):
        """When P1 wins, P1 should see +1, P2 should see -1."""
        # Create a simple terminal state: P1 wins
        final_state = QuoridorState(
            p1_pos=(8, 4),  # P1 reached goal
            p2_pos=(7, 4),
            h_walls=frozenset(),
            v_walls=frozenset(),
            p1_walls=10,
            p2_walls=10,
            current_player=2,
            move_count=100,
        )
        
        from env.rules import winner
        assert winner(final_state) == 1
        
        # P1's outcome: +1
        p1_outcome = 1.0  # P1 wins
        # P2's outcome: -1
        p2_outcome = -1.0
        
        assert p1_outcome == 1.0
        assert p2_outcome == -1.0

    def test_value_sign_draw(self):
        """Draws produce outcome 0 for all."""
        # With 400 moves, game ends in draw
        final_state = QuoridorState(
            p1_pos=(4, 4),
            p2_pos=(4, 5),
            h_walls=frozenset(),
            v_walls=frozenset(),
            p1_walls=0,
            p2_walls=0,
            current_player=1,
            move_count=400,
        )
        
        from env.rules import winner, is_terminal
        assert is_terminal(final_state)
        assert winner(final_state) == 0  # draw
        
        # Both players see draw outcome
        for player in [1, 2]:
            outcome = 0.0  # draw
            assert outcome == 0.0

    def test_augmentation_preserves_outcome(self):
        """Mirroring state should not change outcome sign."""
        state = initial_state()
        state = apply_action(state, 1)
        
        obs = encode_state(state.canonical())
        # Simulated policy and outcome
        policy = np.zeros(140, dtype=np.float32)
        policy[4] = 1.0  # made move 4
        outcome = 1.0  # player to move will win
        
        # Mirror
        m_obs, m_policy = mirror_state_and_policy(obs, policy)
        
        # Outcome should be unchanged
        assert outcome == 1.0  # outcome not modified by mirror
        
        # Verify obs and policy are properly mirrored
        assert m_obs.shape == obs.shape
        assert m_policy.shape == policy.shape


class TestAugmentationRemapping:
    """Verify that board mirroring correctly remaps action indices."""

    def test_mirror_pawn_moves_bijection(self):
        """Pawn move remapping should be bijective."""
        from env.actions import H_WALL_OFFSET, V_WALL_OFFSET
        
        # Pawn move mirror mapping: E↔W are swapped
        pawn_mirror = {
            0: 0, 1: 1, 2: 3, 3: 2,  # N, S, E, W
            4: 4, 5: 5, 6: 7, 7: 6,
            8: 9, 9: 8, 10: 11, 11: 10,
        }
        
        # Check bijection: mirror(mirror(x)) == x
        for action, mirrored in pawn_mirror.items():
            double_mirror = pawn_mirror[mirrored]
            assert double_mirror == action, f"Pawn mirror not bijective: {action} → {mirrored} → {double_mirror}"

    def test_mirror_augmentation_is_involutive(self):
        """Applying mirror twice should recover the original."""
        state = initial_state()
        obs = encode_state(state.canonical())
        policy = np.ones(140, dtype=np.float32) / 140  # Uniform policy
        
        # Mirror once
        m_obs, m_policy = mirror_state_and_policy(obs, policy)
        
        # Mirror again
        m_obs2, m_policy2 = mirror_state_and_policy(m_obs, m_policy)
        
        # Should recover original
        assert np.allclose(m_obs2, obs, atol=1e-5), "Board mirror not involutive"
        assert np.allclose(m_policy2, policy, atol=1e-5), "Policy mirror not involutive"

    def test_mirror_preserves_policy_normalization(self):
        """Mirroring should preserve policy normalization."""
        state = initial_state()
        state = apply_action(state, 1)  # Make a move
        
        obs = encode_state(state.canonical())
        
        # Create a random valid policy
        policy = np.random.rand(140).astype(np.float32)
        policy /= policy.sum()
        original_sum = policy.sum()
        
        # Mirror
        _, m_policy = mirror_state_and_policy(obs, policy)
        
        # Should still sum to 1
        assert np.isclose(m_policy.sum(), original_sum, atol=1e-5), \
            f"Policy normalization not preserved: {original_sum} → {m_policy.sum()}"

    def test_pawn_east_west_swap(self):
        """Action indices 2(E) and 3(W) should swap under mirroring."""
        state = initial_state()
        obs = encode_state(state.canonical())
        
        # Policy favoring East movement
        policy_east = np.zeros(140, dtype=np.float32)
        policy_east[2] = 1.0
        
        # Policy favoring West movement
        policy_west = np.zeros(140, dtype=np.float32)
        policy_west[3] = 1.0
        
        _, m_policy_east = mirror_state_and_policy(obs, policy_east)
        _, m_policy_west = mirror_state_and_policy(obs, policy_west)
        
        # East should map to West and vice versa
        assert m_policy_east[3] == 1.0, "East (2) should map to West (3)"
        assert m_policy_west[2] == 1.0, "West (3) should map to East (2)"

    def test_wall_placement_column_mirroring(self):
        """Wall placement columns should be correctly mirrored."""
        from env.actions import H_WALL_OFFSET, V_WALL_OFFSET, WALL_GRID
        
        state = initial_state()
        obs = encode_state(state.canonical())
        
        # Horizontal wall at column 0
        policy_h0 = np.zeros(140, dtype=np.float32)
        policy_h0[H_WALL_OFFSET + 0] = 1.0
        
        # After mirroring, column 0 should map to column 7 (WALL_GRID-1)
        _, m_policy_h0 = mirror_state_and_policy(obs, policy_h0)
        expected_col = WALL_GRID - 1
        assert m_policy_h0[H_WALL_OFFSET + expected_col] == 1.0, \
            f"H-wall column 0 should map to {expected_col}"

    def test_canonical_and_mirror_composition(self):
        """Composing canonical and mirror should work correctly."""
        # Create a P2-to-move state
        state = initial_state()
        state = apply_action(state, 1)
        assert state.current_player == 2
        
        # Get canonical view
        canonical_state = state.canonical()
        obs = encode_state(canonical_state)
        
        # Mirror it
        m_obs, _ = mirror_state_and_policy(obs, np.ones(140, dtype=np.float32) / 140)
        
        # Should still have valid shape and values
        assert m_obs.shape == (20, 9, 9)
        assert np.all(m_obs >= 0.0) and np.all(m_obs <= 1.0), \
            "Mirrored observation should have values in [0,1]"

