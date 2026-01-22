# Requirements: PixelNav + VLM Planner + Navigation Context (Place–Anchor Memory)

## 0. 문서 목적
본 문서는 AI 코딩 에이전트가 구현 가능한 수준으로 시스템 요구사항을 명확히 정의한다.  
목표는 PixelNav 기반의 저수준 주행 정책은 유지하면서, VLM Planner에 Navigation Context(공간 메모리)를 공급하여

- 반복 주행(oscillation) 및 교착(deadlock)을 감소시키고
- 이미 탐색한 공간의 재방문을 최소화하여
- 장거리 주행 성능(SR, SPL)을 개선하는 것

이다.

---

## 1. 범위(Scope)

### 1.1 포함(In-scope)
- PixelNav 기반 Pixel-Guided Navigation Policy(저수준 정책) 연동
- Planner VLM(**gemini-2.5-flash-lite**, Vertex AI) 기반 상위 의사결정
- Scene 분석용 VLM(Vertex Endpoint 기반)로 place_type(scene_type) 추정
- Place–Anchor Graph Memory(공간 메모리) 생성/갱신
- Discovered Context(DC) FIFO 큐 생성/갱신 및 Planner 입력에 포함
- Action 기반 dead reckoning으로 anchor 좌표 누적 및 간이 localization(근접+의미 기반)

### 1.2 제외(Out-of-scope)
- IMU/휠 인코더 기반 정밀 odometry/SLAM
- 정확한 metric map 구축/loop-closure 최적화(SLAM-level)
- 다중 로봇 협업/지도 공유

---

## 2. 용어 정의(Glossary)
- **Planner VLM**: 파노라마 관측과 Navigation Context를 바탕으로 다음 이동 방향을 결정하는 상위 의사결정 모듈.
- **Pixel-Guided Navigation Policy (PixelNav)**: DINO/SAM 기반 시각적 단서로 waypoint를 선택하고 이산 행동(Action)을 출력하는 저수준 주행 정책.
- **Scene VLM (Scene Analyzer)**: 현재 관측으로부터 scene_type(공간 타입 문자열)을 추정하는 모듈(Vertex Endpoint 호출).
- **Place**: 의미적으로 일관된 공간 단위(방, 복도 등). 시간이 흐르며 요약 정보가 축적됨.
- **Anchor**: 이동 경로상의 위상적 기준점. Place 간 연결 구조와 간이 localization의 최소 단위.
- **Navigation Context**: Planner VLM 입력에 포함되는 메모리 스냅샷. (Place/Anchor 기록 + DCQueue + Localization 상태 + 시스템 프롬프트)
- **Discovered Context (DC)**: Planner VLM이 “다음에 어떤 공간 타입을 목표로 할지/이유”를 짧게 기록한 로그. 반복 주행을 줄이기 위한 negative memory 역할.
- **Oscillation**: 같은 장소를 맴돌거나 A→B→A 형태 반복으로 탐색 진전이 없는 현상.
- **Deadlock**: 탐색이 사실상 정지하거나 동일 행동 반복으로 진행되지 않는 상태.

---

## 3. 구현 상 사실(코드 기준 Traceability)
- Planner VLM 모델명은 **gemini-2.5-flash-lite**를 사용한다.
  - `gpt4v_planner.py` → `gpt_request.py`의 `MODEL_NAME = "gemini-2.5-flash-lite"`를 통해 Vertex AI 호출
- Scene 분석용 VertexAIClient는 코드에서 특정 endpoint_id를 사용한다.
  - `objnav_benchmark.py`에서 endpoint_id를 지정
  - 실제 Gemini 버전은 코드에 직접 표기돼 있지 않으며 **GCP 콘솔의 Vertex Endpoint 설정**을 확인해야 한다.
  - wrapper: `vertex_client_wrapper.py`

---

## 4. 시스템 아키텍처 개요
시스템은 두 개의 루프가 진행된다.

1) **Navigation Loop (상위 의사결정 시점마다 실행)**  
- Input: panoramic image, target object, Navigation Context  
- Output: Action(forward, turn_left, turn_right, stop)

2) **Spatial Memory Update Loop (주기적/이벤트 기반 실행)**  
- Anchor 생성/갱신 (dead-reckoning 기반)  
- 주기적으로 Scene VLM을 호출해 Place 생성/갱신  
- Navigation Context 레코드 업데이트

---

## 5. 전체 시퀀스(Sequence)

### 5.1 Navigation Loop
1. 입력 구성
   - panoramic image (현재 관측)
   - target object (목표 객체 명/클래스)
   - Navigation Context (Place/Anchor + DCQueue + Localization + system prompt)
2. Planner VLM 호출
   - 반환: `goal_flag`, `angle`, `discovered_context`(object)
3. Perception + Policy
   - goal_flag + angle을 DINO & SAM에 전달하여 goal object/관련 마스크/후보를 생성 (10절 참조)
   - Pixel-Guided Navigation Policy가 waypoint를 선택하고 Action을 출력
4. DCQueue 업데이트
   - Planner 출력 `discovered_context`로 DCRecord 1개 생성하여 FIFO 큐에 push (최대 5개 유지)
   - `avoid_hint`는 시스템이 규칙 기반으로 계산하여 DCRecord에 추가(Optional)

### 5.2 Spatial Memory Update Loop
1. Anchor 업데이트
   - Navigation Loop의 Action을 dead reckoning으로 누적하여 (x, y, yaw)을 갱신
   - Anchor 생성 조건을 만족하면 새 anchor를 생성하고 neighbors를 연결
2. Place 업데이트 (주기: 매 20 step)
   - Scene VLM으로 현재 scene_type(place_type 문자열)을 추정
   - place 전환 감지 시 새 Place 생성 또는 기존 Place 갱신
3. Navigation Context 재구성
   - 최신 Place/Anchor 요약, DCQueue, Localization 상태를 prompt에 포함 가능하도록 직렬화

---

## 6. Planner 호출 주기(코드 운영 규칙)
- Planner VLM 호출은 **항상 매 step이 아니다.**
- 기본 규칙:
  1) 에피소드 시작 시 1회 Planner VLM 호출
  2) 이후 low-level policy(PixelNav)가 `STOP`을 출력했지만 **`goal_flag == false`인 경우**, Planner VLM을 1회 재호출하여 새로운 방향/의도를 갱신한다.

(향후 확장 가능: stuck/ABABA 감지 시 재호출 트리거 추가 가능)

---

## 7. 데이터 구조(Data Model)

### 7.1 Place
**정의**: 의미적으로 일관된 공간 영역을 나타내며, 탐색 경험이 누적되는 단위.

- place_id: string/int (유일 식별자)
- place_type(scene_type): string (모델 출력 문자열; 정규화 규칙은 7.3 참조)
- objects_list: set[string]  
  - 누적된 대표 객체/구조 단서 (예: "sofa", "sink", "stove")
- semantic_description: string  
  - place를 요약한 1~2문장 서술(짧게)
- anchors: list[anchor_id]

**생성/갱신 규칙**
- 주기: 매 20 step마다 Scene VLM 호출
- place_type 전환 감지는 “정규화된 scene_type 문자열” 기반으로 판단
- (권장 초기값) dwell_time: 최근 k step 동안 동일 scene_type 유지 시 확정 (k=3)
- 전환 시: 새 Place 생성 + 해당 시점 anchor를 Place의 첫 anchor로 등록

### 7.2 Anchor
**정의**: 이동 경로상에서 생성되는 위상적 기준점. 연결 구조(neighbors)와 간이 localization의 최소 단위.

- anchor_id: string/int
- pose: (x: float, y: float, yaw: float)  # dead reckoning 누적 좌표
- neighbors: list[anchor_id]
- place_id: place_id

**dead reckoning 업데이트 규칙 (Habitat-Sim 설정과 정합)**
- 시작 pose = (0, 0, 0)
- Action을 아래로 해석해 pose 누적
  - forward: +d 만큼 yaw 방향으로 이동 (기본 d=0.25)
  - turn_left: yaw += +30deg
  - turn_right: yaw += -30deg
  - stop: 변화 없음

> NOTE: 본 시스템은 글로벌 GT pose를 사용하지 않으며, Action 기반 누적은 drift를 포함할 수 있다.
> (SLAM 수준의 보정/최적화는 Out-of-scope)

**Anchor 생성 조건(최소 하나 이상 만족 시 생성)**
- 일정 거리: 마지막 anchor 이후 forward 누적이 S step 이상 (기본 S=2)
- 방향 변화: yaw 변화(회전)가 발생했고, 회전 후 forward가 1회 이상 수행됨
- 분기점 이벤트: 문/교차로 등 분기 감지 플래그 true (구현 시 TBD)
- 새로운 Place 진입 시 “첫 anchor”는 반드시 생성

---

## 8. Localization (간이)

### 8.1 배경
- IMU/휠 인코더 odometry를 사용하지 않는다.
- 대신 Action 기반 dead reckoning으로 생성된 anchor 좌표계를 사용한다.

### 8.2 근접 + 의미 기반 “동일 공간 후보” 검색
- 현재 anchor A에서, 모든 anchor B에 대해 dist(A,B)를 계산한다.
- dist(A,B) <= r 이고, `norm_scene_type(A) == norm_scene_type(B)` 이면 “동일 공간 후보”로 간주한다.
- r 기본값: 1.5 (Habitat 단위 기준; 튜닝 가능)

### 8.3 정책
- 본 localization 결과는 **그래프 병합/loop-closure에 사용하지 않는다.**
- 오직 Planner 입력에 “nearby 후보 힌트”로만 제공(Optional)한다.

---

## 9. Navigation Context 정의 및 직렬화

### 9.1 포함 내용
- Place 요약(최근 M개 또는 현재 place 중심)
- Anchor 요약(현재 anchor, 인접 neighbors, 최근 K step anchor 시퀀스)
- Discovered Context Queue(최대 5개, 시간 순서)
- Localization 상태(현재 pose, current_place_id, current_anchor_id, 후보 anchor 목록 요약(Optional))
- system prompt(출력 포맷 강제 및 금지사항)

### 9.2 scene_type / place_type 정규화 규칙(코드 기준)
- Scene VLM/Planner가 반환하는 scene_type 문자열은 **그대로 수용**한다(고정 enum 없음).
- 내부 로직에서는 아래 최소 정규화만 수행한다:
  - 소문자화
  - 마침표/특수문자 제거
  - `"hallway" -> "corridor"` 치환
- 특별 취급: `"corridor" in scene_type` 여부를 별도 조건으로 사용 가능(기존 코드 로직과 정합).

---

## 10. Discovered Context(DC) 상세

### 10.1 정의
DC는 Planner VLM이 현재 관측 + Navigation Context를 바탕으로 내린
- “다음 목표 공간 타입(goal_scene_type)”과
- “그 이유(why)”
를 짧게 기록하는 온라인 컨텍스트 로그이다.

### 10.2 생성 규칙
- Planner VLM 호출 1회당 DCRecord 1개 생성
- DC는 Place/Anchor의 정적 요약(semantic_description, objects_list)을 반복하지 않는다.

### 10.3 최소 필드(Min-DC)
- idx: 1..N (1=가장 오래된, N=가장 최신; 매 호출 시 재부여)
- goal_flag: boolean (현재 관측에서 목표 객체가 보이면 true)
- goal_scene_type: string (Planner가 반환한 문자열; 고정 enum 아님)
- why: string (1문장, 짧게)
- optional avoid_hint: string (규칙 기반)

### 10.4 저장 규칙(Queue)
- FIFO Queue, 최대 크기 N=5
- 새 레코드 push 시 overflow면 가장 오래된 항목 pop
- 매 Planner 호출 시 idx를 1..N으로 재부여
- Planner 입력 시 idx=1→N 순서로 포함

### 10.5 avoid_hint 생성 규칙(규칙 기반)
- avoid_hint는 VLM이 생성하지 않고, 최근 Anchor/Place 시퀀스에서 패턴을 계산해 제공한다.
- 최소 지원:
  - pattern:STUCK  (최근 W step 동안 unique_places <= 1; 기본 W=10)
  - pattern:ABABA  (최근 4개 place가 A,B,A,B 이고 A!=B)

---

## 11. Planner VLM 입출력 규격

### 11.1 입력(요약)
- panoramic image (또는 이미지 참조)
- target object (string)
- Navigation Context (텍스트/JSON 직렬화)

### 11.2 출력(엄격)
Planner VLM은 반드시 아래 JSON만 출력한다(추가 텍스트 금지).

예시:
{
  "goal_flag": true,
  "angle": 15.0,
  "discovered_context": {
    "goal_scene_type": "kitchen",
    "why": "주방은 목표와 연관된 물체가 있을 확률이 높음"
  }
}

- goal_flag: boolean
- angle: float
  - 정의: 파노라마 기준 “정면(0deg)”에서의 상대 방위각(도 단위)
  - 양수=좌회전, 음수=우회전
  - 권장 범위: [-180, 180]
- discovered_context.goal_scene_type: string (모델 출력 문자열)
- discovered_context.why: 1문장

### 11.3 출력 검증/오류 처리
- JSON 파싱 실패, 필드 누락, angle 범위 밖 등의 오류가 발생하면:
  1) **재시도 1회**
  2) 재시도 실패 시 fallback 적용:
     - goal_flag=false
     - angle=0
     - discovered_context.goal_scene_type="corridor"
     - discovered_context.why="fallback"

---

## 12. PixelNav / DINO / SAM 인터페이스(최소 정의, TBD)
- DINO/SAM은 goal_flag/angle을 이용해 “목표 객체 후보” 또는 “waypoint 생성에 필요한 시각적 단서”를 만든다.
- 최소 요구:
  - angle을 이용해 파노라마에서 관심 방향 ROI를 결정하거나,
  - goal object 탐지를 위한 prior로 사용한다.
- 실제 구현은 PixelNav 코드베이스 입력 포맷에 맞춰 통일(TBD).

---

## 13. 로깅/디버깅 요구사항
- 매 step 로그:
  - step_id, action, pose(x,y,yaw), current_place_id, current_anchor_id
  - Planner VLM 호출 여부, 출력 JSON
  - DCQueue(최대 5개)
- 이벤트 로그:
  - place 생성/전환 시점, anchor 생성 시점, ABABA/STUCK 감지 시점
  - fallback 발생 횟수/사유

---

## 14. 평가 지표(Evaluation)
- 기본: SR, SPL (PixelNav baseline 대비)
- Oscillation/Deadlock 보조 지표(최소 1개 이상)
  - ABABA 발생 횟수/에피소드
  - STUCK 발생 횟수/에피소드
  - 최근 W step 동안 unique place 비율

---

## 15. 기본 파라미터(Default Parameters, 초기값)
- forward step distance d = 0.25
- turn angle = 30deg
- place update period = 20 step
- Planner 호출: episode start + (STOP && goal_flag==false)일 때
- DCQueue max size N = 5
- localization radius r = 1.5 (tune)
- stuck window W = 10
- anchor distance threshold S = 2 step
- dwell_time k = 3 (place 전환 확정)

---

## 16. 구현 체크리스트(Implementation Checklist)
- [ ] Navigation Context 직렬화 함수 (요약형)
- [ ] Planner VLM 호출 래퍼 + 출력 JSON 검증기 + retry/fallback
- [ ] DCQueue 자료구조(FIFO, re-index) + avoid_hint 규칙 계산
- [ ] Action -> dead reckoning pose 누적
- [ ] Anchor 생성/연결(neighbors)
- [ ] Scene VLM 호출 래퍼 + scene_type 정규화
- [ ] Place 업데이트(20 step 주기 + 전환 규칙)
- [ ] 간이 localization(근접 + 의미) + 힌트 제공(병합 없음)
- [ ] 로그/재현 가능한 실험 설정(seed, config)
