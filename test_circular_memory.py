
import unittest
from place_graph.circular_memory import CircularMemory, Slot, K_SLOTS

class TestCircularMemory(unittest.TestCase):
    def setUp(self):
        self.mem = CircularMemory()

    def create_mock_slots(self, base_tokens, base_landmarks=None, shift=0):
        slots = []
        if base_landmarks is None:
            base_landmarks = [set()] * K_SLOTS
            
        for i in range(K_SLOTS):
            src_idx = (i - shift) % K_SLOTS
            tokens = set()
            if src_idx < len(base_tokens):
                tokens = base_tokens[src_idx]
            
            landmarks = set()
            if src_idx < len(base_landmarks):
                landmarks = base_landmarks[src_idx]
                
            slots.append(Slot(index=i, tokens=tokens, landmarks=landmarks))
        return slots

    def test_shift_matching(self):
        # Scenario: 6-slot system
        # Slot 0: [sofa]
        # Slot 1: [tv]
        base = [set(["sofa"]), set(["tv"])] + [set()] * 4
        
        # Initial scan
        scan1 = self.create_mock_slots(base, shift=0)
        pid1, shift1, score1, revisit1 = self.mem.assign_place(scan1)
        
        self.assertFalse(revisit1)
        self.assertEqual(shift1, 0)
        
        # Second scan: Same place, rotated by 1 slot (60 degrees)
        scan2 = self.create_mock_slots(base, shift=1)
        pid2, shift2, score2, revisit2 = self.mem.assign_place(scan2)
        
        self.assertTrue(revisit2)
        self.assertEqual(pid1, pid2)
        self.assertEqual(shift2, 1)

    def test_weighted_landmark_matching(self):
        # Place A: Sofa, with red_striped_sofa landmark
        tokens_a = [set(["sofa"])] + [set()] * 5
        landmarks_a = [set(["red_striped_sofa"])] + [set()] * 5
        
        # Place B: Sofa, but with blue_leather_sofa landmark
        tokens_b = [set(["sofa"])] + [set()] * 5
        landmarks_b = [set(["blue_leather_sofa"])] + [set()] * 5
        
        # Initial scan of Place A
        scan_a = self.create_mock_slots(tokens_a, landmarks_a)
        pid_a, _, _, _ = self.mem.assign_place(scan_a)
        
        # Scan of Place B (Different landmarks, same generic tokens)
        scan_b = self.create_mock_slots(tokens_b, landmarks_b)
        # Threshold is 0.35. 
        # Token match = 1.0 (sofa matches sofa)
        # Landmark match = 0.0
        # Weighted score = 0.3 * 1.0 + 0.7 * 0.0 = 0.30
        pid_b, _, score, revisit = self.mem.assign_place(scan_b)
        
        # 0.30 < 0.35, so it should be a NEW place!
        self.assertFalse(revisit, f"Score {score} unexpectedly high")
        self.assertNotEqual(pid_a, pid_b)

    def test_door_memory(self):
        # Scan with a door at slot 2 (mapped from 60 deg slices)
        base = [set()] * 6
        base[2] = set(["door"])
        
        scan1 = self.create_mock_slots(base, shift=0)
        self.mem.assign_place(scan1)
        self.mem.manage_doors(self.mem.current_place_id, scan1)
        
        door_id = scan1[2].door_ids[0]
        self.assertIsNotNone(door_id)
        
        self.mem.update_door_status(self.mem.current_place_id, door_id, "scene_changed", "place_new")
        self.assertTrue(self.mem.door_memory[(self.mem.current_place_id, door_id)].visited)

    def test_door_categorization(self):
        # Place 1: Door at Slot 2
        base1 = [set()] * 6
        base1[2] = set(["door"])
        scan1 = self.create_mock_slots(base1, shift=0)
        
        pid1, _, _, _ = self.mem.assign_place(scan1)
        self.mem.manage_doors(pid1, scan1)
        door_id_1 = scan1[2].door_ids[0]
        
        # Move to Place 2 via Door 1
        # Place 2: Door at Slot 4 (this door leads back to P1)
        base2 = [set()] * 6
        base2[4] = set(["door"])
        scan2 = self.create_mock_slots(base2, shift=0)
        
        pid2, _, _, _ = self.mem.assign_place(scan2)
        self.mem.manage_doors(pid2, scan2)
        door_id_2 = scan2[4].door_ids[0]
        
        # Update door status: P1 Door 1 leads to P2
        self.mem.update_door_status(pid1, door_id_1, "scene_changed", pid2)
        
        # Now get context for P2
        # It should identify Door 2 as [SOURCE/BACKTRACK] because it leads to P1
        context = self.mem.get_nav_context("chair")
        
        self.assertIn("[SOURCE/BACKTRACK]", context)
        self.assertIn(f"AVOID: Slot 4 is the path you just came from ({pid1})", context)

if __name__ == '__main__':
    unittest.main()
