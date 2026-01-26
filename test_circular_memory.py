
import unittest
from place_graph.circular_memory import CircularMemory, Slot, K_SLOTS

class TestCircularMemory(unittest.TestCase):
    def setUp(self):
        self.mem = CircularMemory()

    def create_mock_slots(self, base_tokens, shift=0):
        slots = []
        for i in range(K_SLOTS):
            # Shifted index: slot[i] gets tokens from base[(i-shift)%K]
            # e.g. if shift=3 (90 deg right), then slot[3] sees what was at 0.
            # So slot[i] content should come from base index (i - shift) % K
            src_idx = (i - shift) % K_SLOTS
            tokens = set()
            if src_idx < len(base_tokens):
                tokens = base_tokens[src_idx]
            slots.append(Slot(index=i, tokens=tokens))
        return slots

    def test_shift_matching(self):
        # Scenario: 
        # Slot 0: [sofa]
        # Slot 1: [tv]
        # Others: []
        base = [set(["sofa"]), set(["tv"])] + [set()] * 10
        
        # Initial scan
        scan1 = self.create_mock_slots(base, shift=0)
        pid1, shift1, score1, revisit1 = self.mem.assign_place(scan1)
        
        self.assertFalse(revisit1)
        self.assertEqual(shift1, 0)
        self.assertEqual(score1, 1.0)
        
        # Second scan: Same place, rotated by 3 slots (90 degrees)
        # Scan 2 slot 3 should match scan 1 slot 0
        scan2 = self.create_mock_slots(base, shift=3)
        pid2, shift2, score2, revisit2 = self.mem.assign_place(scan2)
        
        self.assertTrue(revisit2)
        self.assertEqual(pid1, pid2)
        self.assertEqual(shift2, 3) # Should detect 3-slot shift
        self.assertAlmostEqual(score2, 1.0)

    def test_door_memory(self):
        # Scan with a door at slot 5
        base = [set()] * 12
        base[5] = set(["door"])
        
        scan1 = self.create_mock_slots(base, shift=0)
        self.mem.assign_place(scan1)
        
        # Manage doors (assign IDs)
        self.mem.manage_doors(self.mem.current_place_id, scan1)
        
        # Check if ID assigned
        door_id = scan1[5].door_ids[0]
        self.assertIsNotNone(door_id)
        
        # Check visited status (False initially)
        info = self.mem.door_memory[(self.mem.current_place_id, door_id)]
        self.assertFalse(info.visited)
        
        # Update status
        self.mem.update_door_status(self.mem.current_place_id, door_id, "scene_changed", "place_new")
        self.assertTrue(self.mem.door_memory[(self.mem.current_place_id, door_id)].visited)

if __name__ == '__main__':
    unittest.main()
