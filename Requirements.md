content = r"""'''md
# Development Requirement Proposal (Canonical Spec)
## Semantic Circular Memory + Door Experience for Metric‑Free Object Navigation (Single‑Episode)

> **Canonical decision**: **Single‑episode memory**. All memory resets at episode start.  
> **Hard constraint**: **No metric units** (no meters, degrees, radians, global coordinates).  
> **Core idea**: Navigate by **what is seen + remembered**, not by geometry.

---

## 0) What we discussed / decided / lacked (from the conversation)

### 0.1 Decided
1) **Event‑driven planning**
- Run the heavy pipeline only on:
  - **Episode Start**
  - **Re‑planning Event** = PixelNav outputs **STOP** while **goal_flag == False**

2) **Planner pipeline stays**
- **Planner VLM**: (panoramic image + nav context) → **angle_slot & goal_flag**
- **DINO/SAM + PixelNav**: take that direction → execute low‑level actions

3) **Circular Memory purpose**
- Primary: **revisit detection** + **local yaw alignment** (rotation alignment)  
- Implemented via **circular shift matching** between semantic sequences

4) **Semantic sequence representation**
- Use **object name tokens only** (fixed vocabulary)
- **Do NOT** use free‑form attributes (size/color/distance/material/etc.), because they vary with pose/lighting and break matching

5) **Door ID + Door Memory is required**
- To prevent oscillation **before entering a door**, Planner must see door visitation status **in advance**
- Therefore, **door tokens inside the circular list must carry door state**
  - Example: `door(id=2, visited=false)` inside the sequence itself

6) **Visited=true trigger is explicit**
- Mark a door **visited=true** only when **scene transition is confirmed**
  - If we chose a door and, after executing, **SceneType changes** and the change is confirmed (stable), then the chosen door becomes visited
  - If STOP without scene change: not “visited”, just an attempt/failure record

7) **Single‑episode scope**
- Place IDs, Door IDs, DCQueue, Circular Storage, DoorMemory 모두 **episode 종료 시 초기화**

---

### 0.2 Not fully decided (open items)
- Exact object vocabulary list (final fixed set)
- Exact similarity function (how robust / how strict)
- How to handle:
  - repeated objects (chair chair chair)
  - missing detections / hallucinations
  - open spaces with weak “door” cues
  - similar corridors (aliasing)
- SceneType VLM stability rule (k confirmations)

---

### 0.3 Lacked / risks identified (must be documented)
- Object omissions cause mismatch → need thresholding + robust scoring
- SceneType false positives → can incorrectly mark door visited
- Similar-looking regions (corridor aliasing) → revisit mistakes
- Door detection purely from VLM tokens may be unstable → need contract + normalization

---

## 1) Scope & Non‑Goals

### 1.1 Scope (what we build)
- Habitat ObjectNav‑style episode navigation:
  - Start somewhere
  - Goal: find target object (e.g., bed)
  - Use perception + memory to decide waypoint direction repeatedly
- System is **metric‑free**:
  - no global map coordinates
  - no distance/angle values (only discrete indices, tokens, and IDs)

### 1.2 Non‑Goals
- Building a metric map / SLAM / odometry system
- Cross‑episode memory (disallowed)
- Continuous control policy design (PixelNav is assumed existing)

---

## 2) Key Definitions (Terminology)

### 2.1 Discrete direction
- We represent 360° as **K discrete slots**: `slot ∈ {0..K-1}`
- This is NOT a metric angle. It is an index.

### 2.2 Scan (one event snapshot)
- At a trigger event, robot performs a **360 scan**
- Output:
  - panoramic image (for Planner VLM)
  - semantic circular list (for memory + matching)
  - door list with door states embedded

### 2.3 Semantic token
- A token is a single item from a fixed vocabulary, e.g.:
  - `bed`, `lamp`, `wardrobe`, `sofa`, `door`, `toilet`, ...
- No free‑form adjectives.

### 2.4 Circular list / circular sequence
- A list that represents a circle; start index is arbitrary.
- Matching is done by circular shifting.

### 2.5 Place
- A **Place** is an episode‑local ID assigned to a recurring semantic circular pattern.
- A place is NOT a coordinate.

### 2.6 Door
- A **Door** is a special token type that connects places.
- Doors are assigned IDs and tracked to avoid oscillation.

---

## 3) High‑Level Architecture

### 3.1 Modules
1) **Scanner**
- Rotates robot (in sim) to obtain per‑slot RGB observations

2) **Scene VLM (Object Token Extractor)**
- For each slot image, outputs a small list of **object tokens only**

3) **Circular Memory**
- Stores the last N scan circular lists in time order
- Supports circular shift matching for revisit + alignment

4) **SceneType VLM**
- Separate VLM that outputs a coarse scene label (place type) for transition confirmation
- Used primarily to mark door visited

5) **Planner VLM**
- Inputs: panoramic + nav context
- Outputs: `angle_slot`, `goal_flag`, `why`

6) **DINO/SAM**
- Uses `angle_slot` to focus attention/ROI and generate masks/targets

7) **PixelNav Policy**
- Low‑level actions until STOP or termination

8) **DCQueue**
- Stores Planner decisions and outcomes (learning‑style episodic log)

9) **DoorMemory**
- Stores per‑door experience within the episode (visited/attempts/outcomes/optional edge)

---

## 4) Episode Lifecycle (State Machine)

### 4.1 Episode states
- INIT
- SCAN_AND_UPDATE (triggered on Episode Start or Re‑planning)
- PLAN
- EXECUTE
- TERMINATE

### 4.2 Transition rules
- INIT → SCAN_AND_UPDATE (Episode Start)
- SCAN_AND_UPDATE → PLAN
- PLAN → EXECUTE
- EXECUTE → TERMINATE (goal achieved OR max steps)
- EXECUTE → SCAN_AND_UPDATE if (PixelNav STOP && goal_flag == False)

---

## 5) Data Contracts (Strict Interfaces)

### 5.1 Fixed vocabulary
- A single list: `VOCAB = {...}` must be fixed before experiments.
- Scene VLM must output only tokens from VOCAB.
- Any out‑of‑vocab output must be:
  - mapped to `unknown` (optional), or
  - dropped (recommended early stage)

### 5.2 Scene VLM output (per slot)
For each slot image:
- Output: list of tokens (order not critical per slot; order is across slots)

Example per slot:
- `["door", "sofa", "lamp"]`

### 5.3 Scan output (per event)
A scan produces:
- `slot_tokens[i]`: token list for slot i
- `circular_list`: circular representation (defined below)
- `door_tokens`: doors with IDs and states embedded
- `pano_image`: panoramic image for Planner

### 5.4 Planner output (strict JSON)
Required fields:
- `angle_slot`: integer in [0, K-1]
- `goal_flag`: boolean
- `why`: short text (can be stored in DCQueue)

Optional:
- `chosen_door_id`: integer (if Planner explicitly chooses a door)

---

## 6) Core Representation

### 6.1 Slot‑wise representation (recommended)
Instead of a single flat list of tokens, store K slots each with tokens:
- `S = [S0, S1, ..., S(K-1)]`
- Each `Si` is a multiset/list of tokens for that direction.

This makes matching more robust than a single flat sequence.

### 6.2 Door token embedding
Within each slot’s token list, a door token becomes a structured token:

- `door(id=<int>, visited=<bool>, attempts=<int>, last_result=<enum>)`

But to keep Planner input simple:
- Minimal: `door(id=X, visited=true/false)`
- Add more fields later.

---

## 7) Circular Shift Matching (Revisit + Alignment)

### 7.1 Purpose
Given current scan `S_cur` and past scans `S_past`:
- Find best matching past place (if any)
- Compute best circular shift (yaw alignment index)
- Produce a score and revisit decision

### 7.2 Similarity: robust to missing tokens
We do NOT require perfect match.
We compute per-slot similarity and sum across slots.

**Per-slot similarity (Jaccard on token sets)**
- Convert each slot token list to a set (or multiset) excluding door metadata (use just “door” presence for base revisit, or ignore doors entirely for place matching).
- Jaccard:
  - `sim_slot = |A ∩ B| / |A ∪ B|`
- Total:
  - `sim_total = (1/K) * Σ sim_slot`

This survives:
- missing lamp
- extra noise token
- lighting/pose variations (since we only use names)

### 7.3 Best shift computation
For each shift `s` in [0..K-1], compare:
- `S_cur[i]` with `S_ref[(i+s) mod K]`

Pseudo:

    def jaccard(a_set, b_set):
        if len(a_set) == 0 and len(b_set) == 0:
            return 1.0
        return len(a_set & b_set) / max(1, len(a_set | b_set))

    def score_shift(S_ref, S_cur, shift):
        total = 0.0
        for i in range(K):
            A = normalize_tokens(S_ref[i])   # set of tokens
            B = normalize_tokens(S_cur[(i+shift) % K])
            total += jaccard(A, B)
        return total / K

    def best_shift_and_score(S_ref, S_cur):
        best_s, best_score = 0, -1
        for s in range(K):
            sc = score_shift(S_ref, S_cur, s)
            if sc > best_score:
                best_score = sc
                best_s = s
        return best_s, best_score

### 7.4 Place matching across all stored places
We maintain a set of place prototypes, each with one or more representative scans:

    def match_place(place_db, S_cur):
        best = None  # (score, place_id, shift)
        for place_id, reps in place_db.items():
            for S_rep in reps:
                shift, score = best_shift_and_score(S_rep, S_cur)
                if best is None or score > best[0]:
                    best = (score, place_id, shift)
        return best  # may be None if db empty

### 7.5 Revisit decision and place assignment
Let best match be `(score*, place_id*, shift*)`.

Decision rule:

- If `score* >= T_revisit`: revisit=True, assign existing `place_id*`
- Else: revisit=False, create new place_id, register S_cur as representative

Pseudo:

    def assign_place(place_db, S_cur, T_revisit):
        if len(place_db) == 0:
            new_id = new_place_id()
            place_db[new_id] = [S_cur]
            return new_id, 0, 1.0, False

        best = match_place(place_db, S_cur)
        score, pid, shift = best
        if score >= T_revisit:
            # optionally append S_cur as additional representative
            maybe_add_representative(place_db[pid], S_cur)
            return pid, shift, score, True
        else:
            new_id = new_place_id()
            place_db[new_id] = [S_cur]
            return new_id, 0, score, False

---

## 8) Door ID & Door Memory (Anti‑Oscillation)

### 8.1 Goal
Prevent oscillation like:

- A → (door to B) → B → (door back to A) → A → chooses same door again → repeat

We must know **at A, before entering**, which door was already used.

### 8.2 Door identification within a place
Within a place, doors are identified as tokens in slots.
We assign a door_id **episode‑locally** as `(place_id, door_id)`.

### 8.3 Door matching rule (simple, robust, metric‑free)
We do NOT rely on degrees/meters.
We rely on:
- shift alignment from place match
- slot neighborhood context around the door

**Door signature** (context window)
- For a door located at slot `i`, define signature as tokens in window W:

    signature(i) = normalize_tokens(S[i-W .. i+W])  (circular wrap)

Then match doors by signature similarity (Jaccard).

Pseudo:

    def door_signature(S, i, W):
        tokens = set()
        for d in range(-W, W+1):
            tokens |= normalize_tokens(S[(i+d) % K])
        # ensure 'door' is included to anchor
        tokens.add('door')
        return tokens

    def match_or_create_door_id(place_id, S_cur, i_door, door_db, W, T_door):
        sig = door_signature(S_cur, i_door, W)

        # door_db maps (place_id) -> list of (door_id, signature_proto)
        candidates = door_db.get(place_id, [])

        best = None  # (score, door_id)
        for door_id, proto_sig in candidates:
            score = jaccard(proto_sig, sig)
            if best is None or score > best[0]:
                best = (score, door_id)

        if best is not None and best[0] >= T_door:
            return best[1], sig  # reuse id
        else:
            new_door_id = new_door_id_for_place(place_id)
            candidates.append((new_door_id, sig))
            door_db[place_id] = candidates
            return new_door_id, sig

### 8.4 Embedding door state into the circular list
After door_id assignment, augment door tokens in slots:

- `door(id=X, visited=<bool>)`

`visited` comes from DoorMemory.

### 8.5 DoorMemory structure (episode‑local)
DoorMemory key: `(place_id, door_id)`

Fields (minimum):
- `visited: bool`
- `attempts: int`
- `successes: int` (optional)
- `fails: int`
- `last_result: enum {scene_changed, stop_no_change, goal_found, timeout}`
- `leads_to_place_id: optional` (set after transition)

### 8.6 “visited=true” update rule (SceneType confirmed)
This is a must‑have contract.

We maintain an **ActiveDoorAttempt** when Planner chooses a door:

ActiveDoorAttempt:
- start_place_id
- chosen_door_id
- start_scene_type
- status = ACTIVE

During/after EXECUTE:
- run SceneTypeVLM check
- confirm scene transition using stability K_confirm

Pseudo:

    def confirm_scene_change(scene_history, K_confirm):
        # scene_history is recent scene_type labels
        # change is confirmed if the last K_confirm labels are identical and different from start
        if len(scene_history) < K_confirm:
            return False, None
        last = scene_history[-1]
        if all(x == last for x in scene_history[-K_confirm:]):
            return True, last
        return False, None

    def finalize_door_attempt(active_attempt, scene_history, DoorMemory, current_place_id):
        key = (active_attempt.start_place_id, active_attempt.chosen_door_id)
        DoorMemory[key].attempts += 1

        changed, new_scene = confirm_scene_change(scene_history, K_confirm)
        if changed and new_scene != active_attempt.start_scene_type:
            DoorMemory[key].visited = True
            DoorMemory[key].last_result = "scene_changed"
            DoorMemory[key].leads_to_place_id = current_place_id
        else:
            DoorMemory[key].last_result = "stop_no_change"
            DoorMemory[key].fails += 1

**Interpretation**:
- A door is “visited” only if we truly crossed into a different scene type stably.
- Otherwise: attempted but not visited.

This matches the conversation requirement:
> “문을 들어가기 전에 판단해야 한다.  
> 그래서 door token에 visited 상태를 미리 넣는다.  
> visited는 scene change 확인 시점에서 true로 업데이트한다.”

---

## 9) Circular Storage (time‑ordered, last N)

### 9.1 What is stored
Store scans in time order:
- `HistoryScans = [Scan_t-2, Scan_t-1, Scan_t]`

Each Scan contains:
- `place_id`
- `S_slots` (with door tokens and visited flags already embedded)
- `best_shift` (alignment hint relative to matched place prototype)
- `match_score`
- `planner_decision` (angle_slot, goal_flag)
- `outcome` (stop_no_change / scene_changed / goal_found)

### 9.2 Why last 2–3
- Helps Planner avoid repeating failed choice
- Helps detect oscillation patterns even if door visited wasn’t updated yet

---

## 10) DCQueue (Planner decision log + outcome)

### 10.1 Purpose
DCQueue is the Planner’s “learning‑style episode memory”:
- decision + rationale + resulting outcome
- used to discourage repeating obviously bad choices

### 10.2 Entry format (episode‑local)
Fields:
- `place_id`
- `angle_slot`
- `goal_flag`
- `why`
- `result` (stop_no_change / scene_changed / goal_found / timeout)
- optional `chosen_door_id`

---

## 11) Nav Context to Planner (make it maximally intuitive)

Planner must “just read it” and choose.

Recommended nav context structure:

1) **System rules**
- strict output JSON schema
- do not output metric units
- prefer unvisited doors unless contradicted by goal cues

2) **Goal**
- target object name (e.g., “bed”)

3) **Current scan circular list**
- slots 0..K-1, each with tokens
- doors appear as `door(id=, visited=)`

4) **Recent history (last 2–3 scans)**
- for each: circular list + decision + outcome (short)

5) **Localization hints**
- current place_id
- revisit flag and match_score
- best_shift index (optional to show; can be used internally)

---

## 12) Planner → Execution binding (how angle_slot relates to door)

Planner outputs `angle_slot`. We must bind that to a door_id if the slot contains multiple doors or none.

Binding rule (simple):
- If Planner outputs a slot containing a door token → choose that door
- Else choose nearest slot in circular distance that contains a door and is unvisited (tie-break by minimal distance; still metric‑free since it’s index distance)

Pseudo:

    def circular_distance(a, b, K):
        d = abs(a - b)
        return min(d, K - d)

    def choose_door_from_angle(S_slots, angle_slot, DoorMemory, place_id):
        # exact match first
        doors_here = list_doors_in_slot(S_slots[angle_slot])
        if len(doors_here) > 0:
            return doors_here[0]  # or prefer unvisited among them

        # search nearest door slots
        best = None  # (dist, visited_flag, door_id)
        for i in range(K):
            for door_id in list_doors_in_slot(S_slots[i]):
                visited = DoorMemory[(place_id, door_id)].visited
                dist = circular_distance(i, angle_slot, K)
                candidate = (dist, visited, door_id)
                # prefer smaller dist, prefer visited=false
                if best is None:
                    best = candidate
                else:
                    if candidate[0] < best[0]:
                        best = candidate
                    elif candidate[0] == best[0] and candidate[1] == True and visited == False:
                        best = candidate
        return best[2] if best else None

---

## 13) End‑to‑End Main Loop (Pseudo Code)

    def run_episode(env, target_object):
        # ===== episode-local memory reset =====
        place_db = {}        # place_id -> list of representative S_slots
        door_db  = {}        # place_id -> list of (door_id, proto_signature)
        DoorMemory = {}      # (place_id, door_id) -> stats
        DCQueue = []         # list of decisions+outcomes
        HistoryScans = []    # last N scans

        event = "EPISODE_START"

        while not env.done():
            if event in ["EPISODE_START", "REPLAN"]:
                # 1) 360 scan
                pano_image, slot_images = scanner_collect(env, K)

                # 2) Scene VLM per slot -> tokens
                S_slots = []
                for i in range(K):
                    tokens = scene_vlm_tokens_only(slot_images[i], VOCAB)
                    S_slots.append(tokens)

                # 3) Place assignment via circular shift matching
                place_id, shift, match_score, revisit = assign_place(place_db, S_slots, T_revisit)

                # 4) Door ID assignment & embedding visited flags
                S_slots = embed_doors_with_ids_and_state(
                    place_id, S_slots, door_db, DoorMemory,
                    W=door_context_window, T_door=T_door
                )

                # 5) Build nav context (include last 2–3 HistoryScans + DCQueue)
                nav_context = build_nav_context(
                    target_object, S_slots, HistoryScans, DCQueue,
                    place_id=place_id, revisit=revisit, match_score=match_score, shift=shift
                )

                # 6) Planner VLM
                plan = planner_vlm(pano_image, nav_context)  # returns angle_slot, goal_flag, why

                # 7) Choose door_id bound to plan.angle_slot (optional)
                chosen_door_id = choose_door_from_angle(S_slots, plan.angle_slot, DoorMemory, place_id)

                # 8) Execute low-level until STOP or termination condition
                active_attempt = {
                    "start_place_id": place_id,
                    "chosen_door_id": chosen_door_id,
                    "start_scene_type": scenetype_vlm(env.current_view()),
                    "scene_history": []
                }

                outcome = execute_pixelnav_until_stop_or_goal(
                    env, plan.angle_slot, target_object, active_attempt
                )

                # 9) After execution, update DoorMemory using SceneType confirmed
                current_scene = scenetype_vlm(env.current_view())
                active_attempt["scene_history"].append(current_scene)

                # Possibly do a few checks / accumulate history if needed
                # Then finalize
                # Note: current_place_id will be computed next time at scan; for now store partial
                update_door_memory_post_execution(DoorMemory, active_attempt, outcome)

                # 10) Update DCQueue + HistoryScans
                DCQueue.append({
                    "place_id": place_id,
                    "angle_slot": plan.angle_slot,
                    "goal_flag": plan.goal_flag,
                    "why": plan.why,
                    "chosen_door_id": chosen_door_id,
                    "result": outcome
                })
                DCQueue = DCQueue[-MAX_DCQUEUE:]

                HistoryScans.append({
                    "place_id": place_id,
                    "S_slots": S_slots,
                    "revisit": revisit,
                    "match_score": match_score,
                    "shift": shift,
                    "plan": plan,
                    "outcome": outcome
                })
                HistoryScans = HistoryScans[-N_HISTORY:]

                # 11) Decide next event
                if plan.goal_flag == True or outcome == "goal_found":
                    return "SUCCESS"

                if outcome == "stop_no_goal":
                    event = "REPLAN"
                else:
                    # continue executing or trigger replan based on policy; canonical is STOP->REPLAN
                    event = "REPLAN"

            else:
                event = "REPLAN"

        return "FAIL"

---

## 14) Thought Experiment (ASCII) + System Walkthrough + What breaks

### 14.1 Environment (Living, Hallway, Kitchen, Bedroom, Bathroom)
    +-------------------+
    |     BEDROOM       |
    |   [bed] [desk]    |
    |        |          |
    +--------+----------+
             |
    +--------+----------+
    |      HALLWAY      |
    |    door   door    |
    |   +------+        |
    |   |BATH |         |
    |   +------+        |
    +--------+----------+
             |
    +--------+----------+
    |       LIVING      |
    |  sofa  table lamp |
    |  door      door   |
    |     +------+      |
    |     |KITCH|       |
    |     +------+      |
    +-------------------+

Start: Living corner  
Goal: find **bed**

### 14.2 Episode Start (Trigger)
- 360 scan → object tokens per slot → circular list S_slots

Living current scan (simplified):
- Slots contain tokens; doors embedded with ids and visited flags:

Example Planner‑visible representation:
- slot 0: `[sofa, table]`
- slot 1: `[lamp]`
- slot 2: `[door(id=1, visited=false)]`  (to hallway)
- slot 3: `[fridge, stove, door(id=2, visited=false)]` (to kitchen)
- ...

Planner sees target=bed.
He picks `angle_slot` toward hallway door(id=1).

### 14.3 Execute (PixelNav)
- Move until STOP
- SceneType changes from LIVING → HALLWAY (confirmed)
- Therefore:
  - door(id=1).visited = true

### 14.4 Replan in HALLWAY
New scan:
- Detect doors: back to LIVING, to BEDROOM, to BATH
- All visited=false except possibly the one used earlier
Planner picks door to BEDROOM.

Enter BEDROOM:
- New scan contains `bed`
Planner sets goal_flag=true → success.

### 14.5 Oscillation prevention
If agent returns to LIVING and replans:
- door(id=1, visited=true) is visible **before entering**
Planner prefers door(id=2, visited=false) or other unvisited exits.
Thus A↔B bounce is reduced.

---

## 15) Failure Modes & Required Mitigations (Must‑Have in spec)

### 15.1 Object omission / hallucination
Problem:
- VLM may drop `lamp` or hallucinate `chair`.

Mitigation:
- Use Jaccard‑based scoring per slot (not exact match).
- Use threshold `T_revisit` and avoid “exact match” rule.

### 15.2 Duplicate objects
Problem:
- `chair chair chair` indistinguishable.

Mitigation (phase‑1): accept limitation and rely on other tokens.
Mitigation (phase‑2): allow capped counts per token (“multiset bins”), still metric‑free.

### 15.3 SceneType false positives
Problem:
- SceneType changes incorrectly → door incorrectly visited.

Mitigation:
- Confirm change only when stable for `K_confirm` consecutive checks.
- If uncertain, keep door as attempted but not visited.

### 15.4 Similar corridors (aliasing)
Problem:
- Two places look semantically similar → wrong revisit.

Mitigation options:
- Increase K (more slots) for higher resolution
- Require higher T_revisit for revisit assignment
- Store multiple representatives per place
- Use history consistency (last 2–3 scans must agree)

### 15.5 Doors in open spaces
Problem:
- “door” token might not appear.

Mitigation:
- Allow “opening” token in vocab later, but keep token‑only rule
- If no door tokens exist, Planner chooses angle_slot based on object cues; DoorMemory not used.

---

## 16) Implementation Checklist (Engineering)

### 16.1 Must‑have
- [ ] Fixed VOCAB + strict normalization
- [ ] Scanner: K slot images + pano image
- [ ] Scene VLM token extractor (tokens only)
- [ ] Circular shift matching (place assignment)
- [ ] Place DB (episode local)
- [ ] Door detection + door_id assignment + visited embedding
- [ ] DoorMemory update via SceneType change confirmation
- [ ] DCQueue logging (decision + result)
- [ ] Planner prompt builder + JSON validator
- [ ] Replanning loop: STOP & goal_flag false triggers scan+plan

### 16.2 Nice‑to‑have
- [ ] Multi‑representative place prototypes
- [ ] Duplicate token handling (multisets)
- [ ] Aliasing mitigation with history consistency

---

## 17) Final Method Statement (paper‑ready)
We propose a **metric‑free ObjectNav system** that converts 360° observations into a **semantic circular memory** of object tokens. Revisits and yaw alignment are estimated via **circular shift matching**. Navigation decisions are produced by a **Planner VLM** conditioned on the current scan, the last few scan histories, and a **door‑experience memory** embedded directly into the circular list as `door(id, visited)` tokens. Door visitation is updated only upon **confirmed scene transition**, enabling oscillation avoidance **before entering doors**.

---
END (Canonical Spec)
'''"""
out_path = "/mnt/data/Development_Requirement_Proposal_Canonical.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)
out_path
