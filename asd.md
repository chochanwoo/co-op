# CS470 2026 Spring Final Prep Note: Lectures 10-17

요청에는 `CS460`이라고 되어 있었지만, 현재 폴더의 강의자료와 prep note 파일명은 `CS470`이므로 `CS470 2026 Spring Final Prep Note-small-fonts.pdf` 기준으로 정리했다.

자료 기준:

- 10: `10. Reinforcement Learning I-release.pptx`
- 11: `11. Reinforcement_Learning_II-updated.pptx`
- 12: `12-RL for Video-Understanding-release (1).pdf` only
- 13: `13-LLM_based_agents_prompt_optimization-updated-recording-ver2.pptx`
- 14: `14-Neural_Networks-release.pptx`
- 15: `15-backpropagation.pptx`
- 16: `16-Scaling Laws I-release.pptx`
- 17: `17 Intermediate Scaling Laws.pptx`

## 빠른 회독 전략

기말은 T/F, 알고리즘/짧은 답, 토론형 문제가 섞일 수 있으므로 단순 정의보다 다음 네 가지를 우선 암기한다.

1. 차이 비교: MDP vs RL, model-based vs model-free, passive vs active RL, direct evaluation vs TD, tabular vs approximate Q-learning, soft prompt vs hard prompt, Kaplan vs Chinchilla.
2. 핵심 업데이트식: TD, Q-learning, approximate Q-learning weight update, backprop chain rule, scaling law power-law/log-log form, Chinchilla parametric loss.
3. 한계와 조건: Q-learning convergence 조건, exploration/regret tradeoff, non-linear activation 필요성, vanishing/exploding gradient 원인, power-law scaling이 깨지는 regime.
4. 토론형 답안 구조: "무엇을 예측/최적화하는가", "왜 필요한가", "무엇이 한계인가", "어떻게 완화하는가" 순서.

---

## 10. Reinforcement Learning I

### Definition / Basic Idea

Reinforcement Learning, RL은 agent가 environment와 상호작용하면서 reward feedback을 받고, 장기 expected return을 최대화하는 policy를 학습하는 문제다.

기본 흐름:

- 현재 state `s` 관측
- action `a` 선택
- reward `r`와 다음 state `s'` 관측
- 경험 sample `(s, a, r, s')`로 value나 policy를 갱신

목표:

```text
maximize E[G_t], where G_t = R_{t+1} + gamma R_{t+2} + gamma^2 R_{t+3} + ...
```

여기서 `gamma`는 discount factor다. `gamma`가 작으면 가까운 reward를 중시하고, `gamma`가 1에 가까우면 장기 reward를 중시한다.

### RL vs MDP

MDP는 RL의 수학적 기반이다. 차이는 정보와 실행 방식이다.

| 구분 | MDP / Planning | RL |
|---|---|---|
| Transition `T(s,a,s')` | 알고 있음 | 모름 |
| Reward `R(s,a,s')` | 알고 있음 | 모름 |
| 학습 방식 | model을 사용해 offline 계산 | 실제로 action을 해보고 sample로 학습 |
| 대표 알고리즘 | value iteration, policy iteration | TD learning, Q-learning |
| 핵심 어려움 | 계산량, state/action 크기 | exploration, noisy sample, sample efficiency |

시험 답안 포인트: RL은 MDP를 버리는 것이 아니라, MDP 구조는 가정하지만 `T`와 `R`을 모르는 상황에서 경험으로 학습하는 것이다.

### Model-Based Learning

Model-based RL은 경험으로 approximate MDP model을 먼저 학습한 뒤, 그 model을 알고 있다고 가정하고 planning을 수행한다.

절차:

1. 경험에서 transition count를 모은다.
2. `T_hat(s,a,s') = count(s,a,s') / count(s,a)`로 transition을 추정한다.
3. 관측 reward 평균으로 `R_hat(s,a,s')`를 추정한다.
4. 추정된 MDP에서 value iteration 또는 policy iteration을 수행한다.

장점:

- 한 번 model을 만들면 planning 기법을 그대로 쓸 수 있다.
- 관측한 transition 구조를 재사용하므로 direct evaluation보다 sample-efficient할 수 있다.

단점:

- state/action space가 크면 model 저장과 추정이 어렵다.
- model error가 planning 과정에서 누적될 수 있다.
- 실제로 충분히 방문하지 않은 state/action의 model은 부정확하다.

### Model-Free Learning

Model-free RL은 `T`와 `R`을 명시적으로 추정하지 않고, sample transition으로 value나 Q-value를 직접 갱신한다.

핵심 장점:

- model을 저장하지 않아도 된다.
- `Q(s,a)`를 직접 학습하면 action 선택도 model 없이 가능하다.

핵심 단점:

- sample을 많이 필요로 할 수 있다.
- exploration이 부족하면 중요한 state/action을 배우지 못한다.

### Passive RL

Passive RL은 fixed policy `pi`가 주어진 상태에서 그 policy의 value `V^pi(s)`를 평가하는 문제다. agent는 action을 자유롭게 선택하지 않고 주어진 policy를 따라간다.

목표:

```text
learn V^pi(s)
```

중요한 점: passive RL도 offline planning이 아니다. `T`와 `R`을 모르므로 실제 환경에서 policy를 실행하며 경험을 얻는다.

### Direct Evaluation

Direct evaluation은 state를 방문할 때마다 그 이후 실제로 얻은 discounted return을 기록하고, state별 평균을 내는 방식이다.

```text
V(s) = average of observed returns after visiting s
```

장점:

- 직관적이다.
- `T`, `R`이 필요 없다.
- 충분한 sample이 있으면 평균값에 수렴한다.

단점:

- state 사이의 연결 관계를 활용하지 못한다.
- 비슷한 transition을 공유하는 state도 따로따로 배워야 한다.
- sample efficiency가 낮다.

시험 포인트: direct evaluation은 "방문 후 실제 return 전체"를 평균내고, TD는 "한 transition마다 bootstrapping target"으로 갱신한다.

### Temporal Difference Learning

TD learning은 매 transition마다 value를 갱신한다. fixed policy evaluation을 model-free 방식으로 수행한다.

TD target:

```text
r + gamma V(s')
```

TD update:

```text
V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]
```

또는:

```text
V(s) <- (1-alpha)V(s) + alpha(r + gamma V(s'))
```

해석:

- `r + gamma V(s')`: 새 sample이 제안하는 value
- `V(s)`: 기존 추정값
- `alpha`: learning rate
- bracket term: TD error

TD의 장점:

- episode가 끝날 때까지 기다리지 않아도 된다.
- state 연결 구조를 bootstrapping으로 활용한다.
- likely transition은 자주 관측되므로 자연스럽게 더 자주 반영된다.

TD의 한계:

- state value `V(s)`만 알면 model 없이 action 선택을 하기 어렵다.
- optimal control을 위해서는 `Q(s,a)`를 배우는 것이 더 직접적이다.

### Active RL: Exploration vs Exploitation

Active RL에서는 agent가 action을 직접 선택하면서 optimal policy를 학습한다.

핵심 tradeoff:

- exploitation: 현재 가장 좋아 보이는 action을 선택해 reward를 얻음
- exploration: 불확실한 action을 시도해 더 나은 policy를 찾음

exploration이 부족하면 local optimum 또는 잘못된 Q-value에 갇힌다. exploration이 과하면 이미 좋은 policy를 알고도 계속 손해를 본다.

### Q-Learning

Q-learning은 model-free, off-policy control 알고리즘이다. `Q(s,a)`는 state `s`에서 action `a`를 한 뒤 최적으로 행동할 때의 expected return이다.

Q-learning update:

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

구성:

- old estimate: `Q(s,a)`
- sample target: `r + gamma max_a' Q(s',a')`
- learning rate: `alpha`
- TD error: `r + gamma max_a' Q(s',a') - Q(s,a)`

policy 추출:

```text
pi(s) = argmax_a Q(s,a)
```

Q-learning이 off-policy인 이유:

- 실제 행동 policy가 무엇이든, update target은 다음 state에서 greedy optimal action `max_a' Q(s',a')`를 사용한다.
- 즉 "내가 실제로 해온 행동"이 아니라 "할 수 있는 최선"을 학습한다.

수렴 조건:

- 충분한 exploration: 모든 relevant `(s,a)`를 충분히 방문해야 한다.
- learning rate가 적절히 감소해야 한다.
- 너무 빨리 감소하면 학습이 멈추고, 너무 크면 수렴하지 않는다.
- MDP가 stationary해야 한다.

### Lecture 10 시험형 질문 답안 포인트

- Model-based RL의 두 단계: empirical model 학습 후 learned MDP planning. 큰 state space에서는 count/model 저장과 충분한 방문이 어렵다.
- TD가 direct evaluation보다 나은 점: 매 transition에서 bootstrapping update를 하므로 state 연결 정보를 활용한다.
- `V(s)`만으로 policy를 만들 수 없는 이유: action별 결과를 모르므로 model이 없으면 action 비교가 안 된다. `Q(s,a)`는 action 선택에 바로 쓸 수 있다.
- Q-learning 수렴 조건: enough exploration, decaying learning rate, stationary environment.

---

## 11. Reinforcement Learning II

### Q-Learning 복습

RL II는 Q-learning을 더 깊게 다룬다. 핵심은 tabular Q-learning이 모든 `(s,a)` 쌍의 값을 table로 유지한다는 점이다.

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

Q-learning은 off-policy라서 행동 중 suboptimal action을 하더라도, 충분히 탐색하면 optimal Q-value에 수렴할 수 있다.

### Exploration vs Exploitation

exploration은 더 좋은 action을 찾기 위한 정보 수집이고, exploitation은 현재 정보로 가장 좋은 action을 선택하는 것이다.

좋은 RL agent는 단순히 최종 optimal policy를 배우는 것만이 아니라, 배우는 과정에서 낭비하는 reward도 줄여야 한다.

### Epsilon-Greedy

epsilon-greedy는 가장 단순한 forced exploration 방법이다.

```text
with probability epsilon: choose random action
with probability 1 - epsilon: choose argmax_a Q(s,a)
```

장점:

- 구현이 쉽다.
- 모든 action을 어느 정도 탐색할 수 있다.

단점:

- 학습이 끝난 뒤에도 epsilon이 남아 있으면 계속 random action으로 손해를 본다.
- random exploration은 "어디가 불확실한지"를 고려하지 않는다.

보완:

- epsilon을 시간에 따라 감소시킨다.
- exploration function으로 uncertainty-aware exploration을 한다.

### Exploration Function

exploration function은 value estimate `u`와 visit count `n`을 받아 optimistic utility를 반환한다.

아이디어:

```text
less visited action => optimistic value
well visited action => normal estimated value
```

예시 형태:

```text
f(u,n) = R_plus if n < N_e
       = u      otherwise
```

modified Q-update에서는 다음 state의 action을 고를 때 단순 `Q`가 아니라 `f(Q, N)`을 사용한다.

의미:

- 아직 나쁘다고 충분히 확인되지 않은 action을 낙관적으로 평가한다.
- unknown state로 이어지는 경로에도 exploration bonus가 전파될 수 있다.

epsilon-greedy와 비교:

| 항목 | epsilon-greedy | exploration function |
|---|---|---|
| 탐색 방식 | 무작위 | 덜 방문한 곳을 낙관적으로 평가 |
| uncertainty 사용 | 거의 없음 | visit count 사용 |
| 학습 후 행동 | epsilon이 남으면 계속 흔들림 | 충분히 방문하면 bonus 감소 |
| 단점 | 비효율적 탐색 | bonus 설계가 필요 |

### Regret

regret은 학습 과정에서 optimal policy를 따랐다면 얻었을 expected reward와 실제 agent가 얻은 expected reward의 차이를 누적한 값이다.

```text
Regret(T) = sum_t [reward of optimal policy - reward of actual policy]
```

중요한 점:

- 최종적으로 optimal policy를 배워도, 학습 중 많이 실수하면 regret이 클 수 있다.
- "optimal하게 되는 것"과 "optimal하게 배우는 것"은 다르다.
- random exploration은 결국 optimal을 배울 수 있어도 regret이 클 수 있다.

### Approximate Q-Learning

tabular Q-learning은 모든 `(s,a)`를 따로 저장하므로 큰 state space에서 불가능하다. approximate Q-learning은 feature representation으로 state/action을 일반화한다.

feature:

```text
f_i(s,a): state-action pair의 속성
```

예:

- closest ghost까지 거리
- closest food까지 거리
- action이 food에 가까워지는지
- tunnel에 있는지
- ghost 수

linear Q-function:

```text
Q(s,a) = w · f(s,a) = sum_i w_i f_i(s,a)
```

장점:

- 경험을 feature weight 몇 개로 압축한다.
- 보지 못한 state도 feature가 비슷하면 generalization할 수 있다.

단점:

- feature가 같거나 비슷해도 실제 value가 다를 수 있다.
- function approximation error가 생긴다.
- 잘못된 feature는 잘못된 generalization을 만든다.

### Approximate Q-Learning Weight Update

TD error:

```text
delta = [r + gamma max_a' Q(s',a')] - Q(s,a)
```

weight update:

```text
w_i <- w_i + alpha delta f_i(s,a)
```

직관:

- 예상보다 결과가 좋으면 active feature의 weight를 올린다.
- 예상보다 결과가 나쁘면 active feature의 weight를 내린다.
- "나쁜 일이 일어난 이유"를 켜져 있던 feature들에게 blame한다.

### Linear Value Function and Least Squares

linear approximation은 regression으로 볼 수 있다.

- prediction: `w · f(x)`
- target: observed or bootstrapped value
- residual/error: target - prediction

Q-learning update는 online least-squares 스타일로 target에 맞게 weight를 조금씩 조정하는 과정으로 해석할 수 있다.

overfitting 포인트:

- model capacity가 너무 크면 training data에 과적합될 수 있다.
- capacity 제한은 generalization에 도움이 된다.

### Policy Search

policy search는 value를 정확히 예측하는 것보다 reward를 직접 잘 얻는 policy를 찾는 데 초점을 둔다.

왜 필요한가:

- Q-value 자체가 부정확해도 action ordering만 맞으면 좋은 policy가 될 수 있다.
- value modeling 목표와 action selection 목표가 항상 같지 않다.

간단한 방법:

1. 초기 feature weight 또는 policy parameter를 둔다.
2. weight를 조금 올리거나 내린다.
3. sample episode를 돌려 reward가 좋아졌는지 본다.
4. 좋아진 방향으로 hill climbing한다.

한계:

- policy 평가에 많은 episode가 필요하다.
- feature 수가 많으면 탐색 비용이 크다.
- noisy reward 때문에 안정적 비교가 어렵다.

### Lecture 11 시험형 질문 답안 포인트

- Q-learning이 off-policy인 이유: 실제 행동과 무관하게 target에서 greedy max를 사용한다.
- epsilon-greedy vs exploration function: random fixed exploration vs uncertainty/visit-count 기반 optimistic exploration.
- regret: 최종 optimal 여부가 아니라 학습 과정의 누적 손실을 측정한다.
- approximate Q-learning의 tradeoff: generalization과 memory 효율을 얻지만 feature bias와 approximation error를 감수한다.

---

## 12. Reinforcement Learning III: Video Understanding

12번은 현재 폴더에 PPTX가 아니라 PDF로 있다. prep note에는 advanced 표시가 많으므로 깊은 수식보다 큰 흐름과 RL fine-tuning 구조를 잡는 것이 중요하다.

### Video Understanding Task Spectrum

video understanding은 단순 classification에서 interactive video QA와 action grounding까지 확장된다.

대표 task:

- Video classification: 전체 video가 어떤 class인지 분류
- Human-object interaction detection: `<human, interaction, object>` 탐지
- VideoQA: video와 question이 주어졌을 때 answer 생성/선택
- Dense video captioning: 긴 untrimmed video를 time segment별로 요약
- Vision-language-action, VLA: vision-language understanding을 넘어 action prediction/grounding 수행

시험 포인트: video는 image보다 temporal grounding, causal reasoning, long context compression이 중요하다.

### VideoQA / EMNLP23

VideoQA는 다음 형식으로 볼 수 있다.

```text
input: video + question
output: answer
```

요구 능력:

- object detection
- attribute classification
- interaction recognition
- temporal grounding
- action recognition
- causal/temporal reasoning

text-only QA와 다른 점:

- visual evidence를 읽어야 한다.
- temporal order와 event boundary가 중요하다.
- answer는 language model이 생성하지만 evidence는 video에서 온다.

### LongVU

LongVU는 long video-language understanding을 위한 spatiotemporal adaptive compression 아이디어와 연결된다.

핵심 필요성:

- long video는 frame/token 수가 너무 많다.
- 모든 frame을 그대로 LLM/MLLM에 넣으면 compute와 memory가 폭증한다.
- 중요한 spatiotemporal information을 보존하면서 압축해야 한다.

시험형 답안:

> Long video understanding에서는 긴 시간 범위의 event를 보존해야 하므로 단순 frame sampling보다 adaptive compression이 중요하다. 목표는 irrelevant/redundant visual token을 줄이면서 temporal reasoning에 필요한 evidence를 남기는 것이다.

### REFT: Reinforcement Fine-Tuning

ReFT는 reasoning이나 answer quality를 reward/feedback으로 개선하기 위해 model을 RL 방식으로 fine-tuning하는 접근이다.

기본 구조:

1. model이 response를 생성한다.
2. reward model 또는 rule-based reward가 response를 평가한다.
3. policy gradient 계열 방법으로 높은 reward response의 확률을 높인다.

왜 필요한가:

- supervised fine-tuning만으로는 reasoning quality, format compliance, preference alignment를 충분히 맞추기 어렵다.
- reward가 task-specific metric이나 human preference를 직접 반영할 수 있다.

### PPO

PPO, Proximal Policy Optimization은 policy update가 너무 커져서 학습이 불안정해지는 것을 막기 위해 clipped objective를 사용한다.

핵심 ratio:

```text
r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
```

clipped objective 직관:

```text
maximize min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)
```

의미:

- advantage가 positive이면 해당 action 확률을 올리고 싶다.
- 하지만 ratio가 너무 커지면 clip해서 과도한 update를 막는다.
- 안정적인 RLHF/RL fine-tuning에 널리 쓰인다.

### GRPO

GRPO, Group Relative Preference Optimization은 같은 prompt에 대해 여러 response를 생성하고, group 내 reward를 비교해 advantage를 만든다.

절차:

1. 같은 input/query에 대해 여러 response를 sample한다.
2. reward model 또는 rule-based metric으로 각 response를 평가한다.
3. group reward를 normalize해서 relative advantage를 계산한다.
4. reward가 높은 response의 확률을 높이는 방향으로 policy를 update한다.

PPO와 비교:

| 항목 | PPO | GRPO |
|---|---|---|
| 기준 | value/advantage estimate 사용 | group 내 relative reward 사용 |
| 장점 | 안정적인 clipped update | 여러 response 비교로 reward baseline 구성 |
| 한계 | value model 비용/불안정성 | reward가 비슷하면 advantage가 약해짐 |

### DeepVideo-R1

DeepVideo-R1은 video reinforcement fine-tuning에서 GRPO의 한계를 다룬다.

문제:

- safeguard가 없으면 RL 학습이 불안정할 수 있다.
- 너무 쉽거나 너무 어려운 sample에서는 group rewards가 비슷해져 advantage가 거의 0이 된다.
- advantage가 0에 가까우면 effective learning signal이 약해진다.

해결 방향:

- Reg-GRPO: advantage maximization을 advantage regression 관점으로 재구성해 implicit reward와 실제 reward-derived advantage를 align한다.
- difficulty-aware data augmentation: group mean reward와 dynamic global mean을 비교해 sample difficulty를 추정하고, 너무 쉽거나 어려운 sample을 적절한 난이도로 조정한다.

시험 답안 포인트:

> DeepVideo-R1의 핵심은 video reasoning을 RL로 강화하되, GRPO의 vanishing advantage 문제를 완화하기 위해 difficulty-aware sample 조정과 regressive objective를 사용하는 것이다.

### Token Merging

Token merging은 visual transformer에서 중복 visual token을 병합해 training/inference를 빠르게 하는 방법이다.

효과:

- memory 감소
- training speed-up
- redundancy 제거
- 경우에 따라 accuracy 개선

video/MLLM에서 중요한 이유:

- video는 token 수가 매우 많기 때문에 token efficiency가 성능과 비용 모두에 중요하다.

### VidChain

VidChain은 dense video captioning을 chain-of-tasks로 나누어 처리한다.

문제:

- baseline은 너무 많은 segment를 만들거나 반복적인 caption을 생성할 수 있다.
- key event detection에 실패하면 전체 story가 흐트러진다.

CoTasks 예:

1. segment 수 예측
2. time boundary 예측
3. 각 segment caption 생성

M-DPO:

- DPO는 preferred response의 likelihood를 dispreferred보다 높인다.
- M-DPO는 METEOR, CIDEr, IoU 같은 metric preference를 사용해 preference pair를 만든다.
- task별 metric gap이 충분히 큰 sample을 사용해 learning signal을 강화한다.

### Magma / VLA

Magma는 vision-language-action model 흐름과 연결된다.

핵심 challenge:

- UI navigation, robot manipulation, video understanding은 input/output 형식이 다르다.
- action-focused robot data가 부족하다.

방법:

- Set-of-Mark: actionable object에 번호를 붙여 coordinate prediction을 번호 선택 문제로 바꾼다.
- Trace-of-Mark: object/hand trajectory를 예측하게 해 temporal/action reasoning을 학습한다.
- video data를 활용해 human-object interaction과 action signal을 보완한다.

---

## 13. LLM-Based Agents and Prompt Optimization

### Prompt Engineering

Prompt engineering은 LLM application에서 prompt를 testing, evaluation, analysis, optimization을 통해 체계적으로 개선하는 practice다.

중요한 이유:

- LLM performance는 prompt wording, examples, output format, role instruction에 민감하다.
- 작은 문구 차이가 accuracy와 reasoning behavior를 크게 바꿀 수 있다.

### Emergent Abilities

GPT-3 이후 large language model에서 few-shot learning, in-context learning, chain-of-thought reasoning 같은 능력이 model scale과 함께 관찰되었다.

emergent ability를 설명할 때 주의할 점:

- model size와 data scale이 커지며 새로운 behavior가 나타나는 것처럼 보일 수 있다.
- 일부 metric은 threshold를 넘으면 갑자기 좋아지는 것처럼 보인다.
- 실제 underlying loss는 smooth하게 개선되지만, task metric은 non-linear하게 보일 수 있다.

시험 답안:

> Emergent ability는 model scale 증가와 함께 in-context examples나 reasoning prompt를 활용하는 능력이 관찰되는 현상이다. 다만 모든 능력이 진정한 discontinuity로 생긴다고 단정하면 안 되고, metric threshold와 evaluation 방식 때문에 abrupt하게 보일 수도 있다.

### In-Context Learning

In-context learning, ICL은 gradient update 없이 prompt 안의 examples/context만으로 task를 수행하는 능력이다.

few-shot prompting:

```text
example 1
example 2
...
query
```

특징:

- model parameter는 바뀌지 않는다.
- task specification이 prompt context에 들어간다.
- example selection과 ordering에 민감하다.

한계:

- 복잡한 multi-step reasoning task에서는 few-shot examples만으로 부족할 수 있다.
- prompt length, example quality, model scale에 의존한다.

### Chain-of-Thought, Few-Shot CoT, Zero-Shot CoT

Chain-of-thought, CoT는 intermediate reasoning steps를 prompt 또는 output에 포함해 reasoning을 유도하는 방법이다.

Few-shot CoT:

- reasoning example을 몇 개 제공한다.
- model이 비슷한 reasoning trace를 따르도록 유도한다.

Zero-shot CoT:

- example 없이 "Let's think step by step" 같은 instruction만 추가한다.
- 일반 zero-shot보다 성능이 크게 좋아질 수 있지만, manual CoT examples보다는 약할 수 있다.

핵심 비교:

| 방식 | 입력 | 장점 | 한계 |
|---|---|---|---|
| few-shot | input-output examples | 간단하고 강함 | reasoning task에는 부족 |
| few-shot CoT | reasoning examples | multi-step reasoning 강화 | example 작성 비용 |
| zero-shot CoT | reasoning instruction | 쉽고 범용적 | prompt wording에 민감 |

### Soft Prompt Learning

soft prompt learning은 사람이 읽을 수 있는 discrete text prompt가 아니라 learnable vector prompt를 최적화한다.

#### CLIP

CLIP은 image와 text를 joint embedding space에 맞추는 vision-language model이다.

핵심:

- contrastive learning
- matching image-text pair는 가깝게
- non-matching pair는 멀게

CLIP classification은 class name을 text prompt로 만들고, image embedding과 text embedding similarity를 비교한다.

#### CoOp

CoOp, Context Optimization은 prompt context를 learnable vectors로 모델링한다.

의미:

- "a photo of a [class]" 같은 hand-crafted template 대신 context token embedding을 학습한다.
- few-shot image classification에서 handcrafted prompt보다 좋은 성능을 낼 수 있다.

장점:

- prompt engineering 자동화
- task-specific adaptation

한계:

- task overfitting
- base-to-new generalization 문제

#### RPO

RPO, Read-only Prompt Optimization은 learnable prompt가 internal representation을 크게 흔들어 generalization이 나빠지는 문제에서 출발한다.

핵심 아이디어:

- masked attention 기반 read-only mechanism
- special token initialization
- pairwise scoring function

목표:

- parameter efficiency
- domain generalization
- base-to-new generalization
- training dataset에 따른 variance 감소

#### DAPT

DAPT, Distribution-Aware Prompt Tuning은 CLIP의 visual/text embedding misalignment 문제를 다룬다.

문제:

- zero-shot CLIP에서 text와 visual modality가 잘 정렬되지 않을 수 있다.
- 같은 class image embedding이 넓게 흩어질 수 있다.
- class 간 embedding이 충분히 분리되지 않을 수 있다.

방법:

- text prompt: inter-dispersion 증가, class text embedding 간 거리 확대
- visual prompt: intra-dispersion 감소, 같은 class image embedding을 cluster

#### ProMetaR

ProMetaR은 prompt learning의 task overfitting을 줄이기 위해 meta-regularization을 사용한다.

구조:

- inner loop: training data로 task-adapted prompt 학습
- outer loop: validation data 성능을 기준으로 regularization 강도를 학습

핵심:

- task-specific knowledge와 task-agnostic general knowledge를 균형 있게 사용한다.
- validation performance를 통해 prompt가 generalize되도록 regularizer를 meta-learn한다.

### LLM-Based Hard Prompt Optimization

hard prompt optimization은 사람이 읽을 수 있는 discrete instruction, examples, context를 최적화한다.

#### TextGrad

TextGrad는 numerical gradient 대신 natural-language critique를 gradient처럼 사용하는 프레임워크다.

구조:

- variable: prompt, code, plan 같은 text
- function: LLM call, tool, simulator 등 black-box component
- loss/evaluator: output을 critique
- backward: critique를 upstream variable에 textual gradient로 전달
- step: LLM optimizer가 variable을 rewrite

시험 답안:

> TextGrad는 non-differentiable compound AI system에서 backprop을 직접 적용할 수 없으므로, LLM critic이 생성한 textual feedback을 gradient처럼 전파해 prompt나 text variable을 업데이트한다.

#### Genetic Algorithm

genetic algorithm은 candidate population을 반복적으로 개선하는 evolutionary search다.

loop:

1. initialize candidates
2. evaluate fitness
3. select strong candidates
4. crossover
5. mutate
6. repeat

prompt optimization에서는 prompt가 candidate solution이고, validation score가 fitness가 된다.

#### GEPA

GEPA는 reflective prompt evolution이다.

세 가지 insight:

- Reflection: numeric score만 보지 않고 domain-specific textual feedback을 learning signal로 사용한다.
- Generic: 좋은 prompt는 기존 좋은 prompt에서 파생된다고 보고 prompt tree를 만든다.
- Pareto: aggregate score만 보지 않고 individual data point별 개선을 추적한다.

algorithm 요약:

1. candidate pool과 validation item별 best prompt, 즉 Pareto front를 유지한다.
2. 개선할 prompt를 선택한다.
3. minibatch rollout에서 score와 textual feedback을 얻는다.
4. LLM이 mutation/crossover로 prompt alternative를 만든다.
5. validation score로 pool을 갱신한다.

장점:

- 적은 rollout budget에서도 효율적일 수 있다.
- 개별 test-case 특화 개선을 나중에 aggregate할 수 있다.

#### PRESTO

PRESTO는 black-box LLM을 위한 instruction optimization에서 white-box LLM과 soft prompt space를 활용한다.

문제:

- discrete instruction search는 어렵다.
- white-box LLM에서 여러 soft prompt가 같은 instruction으로 mapping되는 many-to-one preimage가 있다.
- 같은 instruction을 중복 query하면 black-box call이 낭비된다.

핵심 방법:

- preimage structure를 미리 계산한다.
- score sharing: 한 soft prompt의 score를 같은 preimage의 다른 soft prompt에도 공유한다.
- preimage-based initialization: search space를 넓게 덮도록 초기점을 선택한다.
- score consistency regularization: 같은 preimage 내 score prediction이 일관되도록 한다.

### System Prompts for LLM Agent Systems

system prompt는 agent의 role, behavior, constraints, reasoning protocol, action format, tool-use rule, collaboration rule을 정의한다.

중요한 이유:

- agent system에서는 단순 answer instruction보다 action loop와 tool protocol이 중요하다.
- system prompt가 "agent program"처럼 작동할 수 있다.

#### ReAct

ReAct는 reasoning과 acting을 하나의 loop로 결합한다.

형식:

```text
Thought -> Action -> Observation -> Thought -> ...
```

의미:

- Thought: 다음에 무엇을 할지 reasoning
- Action: search, lookup, tool call 등 외부 환경과 상호작용
- Observation: action 결과

takeaway:

> ReAct의 system prompt는 LLM을 reasoning하고 tool을 사용하며 observation으로 plan을 수정하는 agent protocol로 만든다.

#### CAMEL

CAMEL은 role-playing multi-agent framework다.

구조:

- AI User: planner/instructor 역할
- AI Assistant: executor 역할
- inception prompt로 role, task, communication rule, stopping criterion을 정의한다.

장점:

- 복잡한 task를 step-by-step instruction과 execution으로 분해한다.
- single-shot보다 구체적이고 실행 가능한 solution을 만들 수 있다.

위험:

- role flipping
- 반복 instruction
- task completion criterion 불명확

#### MetaSPO

MetaSPO는 system prompt optimization을 bi-level meta-learning으로 본다.

- inner loop: task-specific user prompt 최적화
- outer loop: 다양한 task feedback으로 shared system prompt 최적화

목표:

- 특정 task가 아니라 다양한 unseen task에 robust한 system behavior를 학습한다.

#### SPRIG

SPRIG는 edit-based genetic algorithm으로 system prompt를 최적화한다.

방법:

- candidate system prompt 생성
- edit, mutation, crossover
- diverse task에서 성능이 좋은 prompt 선택

### Prompts for Multi-Agent Systems

multi-agent research system의 핵심 구조는 orchestrator-subagent다.

- lead agent가 문제를 나눈다.
- subagent가 각자 context와 tool을 사용해 parallel research를 한다.
- lead agent가 결과를 synthesize한다.

prompting principles:

- agent 입장에서 step-by-step으로 실패 가능성을 시뮬레이션한다.
- orchestrator에게 delegation rule을 명확히 준다.
- subagent objective, output format, tool boundary를 분명히 한다.
- 단순 query는 agent 수를 줄이고, 복잡한 open-ended query는 넓게 탐색한다.
- broad search 후 narrow down한다.
- parallel tool calling으로 time을 줄인다.

tradeoff:

- multi-agent는 breadth와 parallelism을 얻는다.
- 대신 token cost와 coordination complexity가 커진다.

---

## 14. MLP

### Perceptron

perceptron은 input feature에 weight를 곱하고 bias를 더한 뒤 activation으로 output을 만드는 가장 기본적인 neural unit이다.

```text
z = w^T x + b
y = activation(z)
```

single-layer perceptron은 linear decision boundary만 표현할 수 있다.

### Linear / Fully Connected / Dense Layer

fully connected layer는 input vector의 모든 component가 output unit들과 연결되는 layer다.

```text
h = W x + b
```

shape 예시:

- input dimension: `d`
- hidden units: `h`
- output classes: `c`

parameter count:

```text
first layer weights: d * h
first layer biases: h
second layer weights: h * c
second layer biases: c
total with bias = d*h + h + h*c + c
```

슬라이드 예시:

```text
input 3072, hidden 100, output 10
weights only = 3072*100 + 100*10 = 308,200
with biases = 308,200 + 100 + 10 = 308,310
```

### Activation Functions and Necessity

neural network가 non-linear activation을 필요로 하는 이유:

```text
linear(linear(linear(x))) = one linear transform
```

즉 activation이 없으면 여러 layer를 쌓아도 결국 하나의 linear classifier와 동일하다. non-linear activation이 있어야 XOR 같은 비선형 decision boundary와 complex function을 표현할 수 있다.

### Activation Function Comparison

| Activation | Form | 장점 | 단점 |
|---|---|---|---|
| sigmoid | `1/(1+e^-x)` | probability-like output, binary output에 직관적 | saturation, vanishing gradient, not zero-centered |
| tanh | `tanh(x)` | zero-centered, sigmoid보다 hidden layer에 나음 | saturation, vanishing gradient |
| ReLU | `max(0,x)` | sparse activation, active region gradient가 큼, deep net 학습에 좋음 | dead ReLU, negative region gradient 0 |
| Leaky ReLU | `max(ax,x)` | negative side에도 작은 gradient | slope 선택 필요 |
| ELU | negative side smooth saturation | mean activation을 음수 쪽으로 보정 가능 | 계산 비용, hyperparameter |
| softmax | `exp(z_i)/sum_j exp(z_j)` | multi-class probability distribution | large logit overflow 주의, 보통 cross-entropy와 함께 사용 |

### Dimensionality and Parameter Counting

시험에서 shape 문제가 나오면 다음 순서로 푼다.

1. input vector dimension 확인
2. 각 layer weight matrix shape 확인
3. bias 포함 여부 확인
4. activation은 parameter가 없는 경우가 대부분임을 기억

예:

```text
x: R^d
W1: R^{h x d}
b1: R^h
h1 = ReLU(W1 x + b1)
W2: R^{c x h}
b2: R^c
logits = W2 h1 + b2
```

### Hyperparameters

MLP에서 중요한 hyperparameters:

- learning rate
- number of layers
- hidden units per layer
- activation function
- regularization strength
- batch size
- initialization

decision boundary 관련 포인트:

- hidden unit과 non-linearity가 많아질수록 더 복잡한 boundary를 표현할 수 있다.
- 하지만 data가 작으면 overfitting 위험이 커진다.
- regularization, early stopping, validation set이 필요하다.

### Mini-Lab / Training NNs on the Web

TensorFlow Playground 같은 미니랩에서 봐야 할 것은 "하이퍼파라미터를 바꾸면 decision boundary와 generalization이 어떻게 바뀌는가"다.

확인 포인트:

- learning rate가 너무 크면 loss가 진동하거나 발산한다.
- hidden layer/unit이 많아지면 복잡한 boundary를 만들 수 있지만 overfitting 위험이 커진다.
- activation function에 따라 학습 속도와 표현력이 달라진다.
- regularization을 키우면 boundary가 smoother해지고 overfitting이 줄 수 있지만, 너무 크면 underfitting된다.

### Lecture 14 시험형 질문 답안 포인트

- MLP와 single-layer perceptron 차이: hidden layer와 non-linear activation을 통해 non-linear function을 표현한다.
- activation 제거 시: 여러 affine transform의 합성은 affine transform 하나로 collapse한다.
- sigmoid/tanh/ReLU 비교: saturation과 gradient flow, zero-centered 여부, dead neuron 위험 중심으로 답한다.
- "optimal architecture"는 고정 답이 없고 data/task/compute에 따라 validation과 regularization으로 선택한다.

---

## 15. Backpropagation

### Backpropagation

Backpropagation은 computational graph에서 chain rule을 사용해 loss의 gradient를 각 parameter에 효율적으로 계산하는 알고리즘이다.

training pipeline:

1. task, performance metric, experience/data 정의
2. model 선택
3. loss function과 regularizer 선택
4. parameter optimization
5. test/evaluation

### Computational Graph

computational graph는 복잡한 함수를 작은 operation node로 나눈 그래프다.

forward pass:

- input에서 output/loss까지 값을 계산한다.
- intermediate value를 저장한다.

backward pass:

- loss에서 input 방향으로 gradient를 전달한다.
- 각 node는 local gradient와 upstream gradient를 곱해 downstream gradient를 만든다.

### Chain Rule

단순 chain:

```text
y = f(x)
L = g(y)
dL/dx = dL/dy * dy/dx
```

여러 변수:

```text
z = f(x, y)
dL/dx = dL/dz * dz/dx
dL/dy = dL/dz * dz/dy
```

backprop의 핵심은 전체 미분을 직접 전개하지 않고, graph node별 local derivative를 재사용한다는 점이다.

### Upstream / Local / Downstream Gradient

용어:

- upstream gradient: 위쪽/loss 쪽에서 들어오는 gradient, 예: `dL/dy`
- local gradient: 현재 operation의 미분, 예: `dy/dx`
- downstream gradient: input 쪽으로 전달할 gradient, 예: `dL/dx`

공식:

```text
downstream = upstream * local
```

자주 나오는 gate:

- add gate: gradient를 그대로 양쪽으로 분배
- multiply gate: 상대 입력값을 곱해서 전달
- max gate: forward에서 선택된 branch로만 gradient 전달
- sigmoid: derivative `sigma(x)(1-sigma(x))`
- ReLU: `x > 0`이면 1, `x <= 0`이면 0

### Backprop for a Simple Neuron

neuron:

```text
z = w^T x + b
a = activation(z)
L = loss(a, y)
```

backprop 순서:

1. `dL/da` 계산
2. activation local gradient로 `dL/dz` 계산
3. `z = w^T x + b`에 대해 `dL/dw`, `dL/db`, `dL/dx` 계산

예:

```text
dL/dw_i = dL/dz * x_i
dL/db = dL/dz
dL/dx_i = dL/dz * w_i
```

### Why Training Mode Requires More Memory than Eval Mode

training mode는 backward pass를 위해 forward pass의 intermediate activation과 computational graph를 저장해야 한다.

필요한 것:

- layer별 activation
- local gradient 계산에 필요한 값
- dropout mask
- batch norm statistics
- optimizer state, 예: momentum, Adam moments

eval/inference mode:

- gradient가 필요 없다.
- activation을 backward용으로 저장하지 않아도 된다.
- dropout은 꺼지고 batch norm은 running statistics를 사용한다.

따라서 training이 inference보다 memory를 훨씬 많이 쓴다.

### Vanishing and Exploding Gradients

깊은 network에서는 gradient가 여러 Jacobian/local gradient의 곱으로 전달된다.

vanishing gradient:

- local derivative가 반복적으로 1보다 작으면 gradient가 0에 가까워진다.
- 앞쪽 layer가 거의 학습되지 않는다.
- sigmoid/tanh saturation에서 자주 발생한다.

exploding gradient:

- local derivative나 weight norm이 반복적으로 1보다 크면 gradient가 폭증한다.
- update가 불안정하고 loss가 NaN이 될 수 있다.

완화 방법:

- ReLU, Leaky ReLU 같은 activation 사용
- Xavier/He initialization
- residual connection
- normalization, 예: batch norm, layer norm
- gradient clipping
- 적절한 learning rate
- RNN에서는 LSTM/GRU 같은 gated architecture

### Activation Function and Gradient Flow

activation은 forward expressivity뿐 아니라 backward gradient flow도 결정한다.

- sigmoid: output이 0 또는 1에 가까우면 derivative가 거의 0이라 gradient가 사라진다.
- tanh: zero-centered라 sigmoid보다 낫지만 saturation은 여전히 문제다.
- ReLU: positive region에서는 gradient가 1이라 vanishing을 줄인다. negative region에서는 gradient가 0이라 dead ReLU가 생길 수 있다.
- Leaky ReLU/ELU: negative region에도 gradient를 남겨 dead neuron 문제를 완화한다.

### Lecture 15 시험형 질문 답안 포인트

- chain rule이 필요한 이유: deep network는 composite function이므로 loss가 각 parameter에 미치는 영향을 node별 local derivative 곱으로 계산해야 한다.
- training memory가 큰 이유: backward 계산을 위해 activation과 graph를 저장해야 하기 때문이다.
- vanishing/exploding 원인: 깊은 layer에서 Jacobian/local gradient가 반복 곱해지는 구조.
- batch size와 gradient estimate: 큰 batch는 variance가 낮지만 compute/memory가 크고, 작은 batch는 noisy하지만 regularization 효과와 빠른 update가 있다.

---

## 16. Introduction to Scaling Laws

### Motivation

large model training은 비용이 크고 반복하기 어렵다. scaling law는 작은 규모 실험에서 큰 규모 성능을 예측해 resource allocation을 돕는다.

비유:

- rocket science처럼 full-scale experiment를 매번 할 수 없다.
- 작은 실험으로 trend를 추정하고 큰 scale로 extrapolate한다.

LLM training에서 중요한 이유:

- GPU/time budget이 제한되어 있다.
- full training 실패는 되돌리기 어렵다.
- model size, data size, compute budget을 어떻게 배분할지 결정해야 한다.

### Definition

scaling law는 training resource가 증가할 때 loss/error가 어떻게 변하는지 설명하는 경험적 법칙이다.

resources:

- model parameters `N`
- dataset size/training tokens `D`
- compute `C`

대표 power-law form:

```text
L(x) = L_infinity + A x^{-alpha}
```

`L_infinity` 또는 `E`는 irreducible loss/error floor로 볼 수 있다.

### Power-Law Form and Log-Log Space

loss floor를 뺀 power-law:

```text
L(x) - L_infinity = A x^{-alpha}
```

log를 취하면:

```text
log(L(x) - L_infinity) = log A - alpha log x
```

따라서 log-log plot에서 직선이 나오면 power-law scaling을 의심할 수 있다.

해석:

- slope: `-alpha`
- alpha가 클수록 resource 증가에 따른 loss 감소가 빠르다.
- 직선성이 좋으면 extrapolation 가능성이 커진다.

### Single-Resource vs Joint Scaling

single-resource scaling:

- 하나의 resource만 변화시킨다.
- 예: parameter `N`만 늘리고 다른 조건은 고정.
- log-log plot에서 직선으로 분석하기 쉽다.

joint scaling:

- `N`, `D`, `C`가 함께 변한다.
- resource 간 interaction을 본다.
- compute-optimal allocation 문제와 연결된다.

### History Before Scaling Laws

초기 learning theory는 sample size에 따른 worst-case bound를 다뤘지만, modern scaling law처럼 실제 neural network 성능을 예측하는 경험적 extrapolation과는 다르다.

pre-formalization evidence:

- SVM, NLP data scaling 등에서 data가 커질수록 performance가 예측 가능하게 개선됨.
- Hestness et al.은 deep learning learning curve가 power-law를 따른다는 경험적 증거를 제시.

### Why Do We Observe a Power Law?

toy example: sample mean estimation.

sample mean의 MSE:

```text
MSE ~= sigma^2 / n
```

이는 `n^{-1}` 형태의 power law다.

일반화된 직관:

- independent sample이 늘면 uncertainty가 감소한다.
- model/data/compute scale이 커질수록 error가 diminishing return 형태로 줄어든다.
- log-log space에서는 이 감소가 직선처럼 보일 수 있다.

### Power-Law Scaling and Empirical Results

#### Feed-Forward Ratio

Kaplan 계열 결과에서는 feed-forward ratio 변화에 performance가 비교적 weakly sensitive했다. 즉 model parameter count나 compute에 비해 세부 architecture ratio의 영향이 작게 관찰될 수 있다.

#### Aspect Ratio

depth/width aspect ratio도 일정 범위 안에서는 loss에 큰 영향을 주지 않을 수 있다. 같은 parameter count라면 여러 depth-width 조합이 비슷한 performance를 낼 수 있다.

#### Attention Head Dimension

attention head configuration 역시 loss에 큰 영향을 주지 않거나, 약간의 compute로 보상 가능하다고 소개된다.

주의: "항상 중요하지 않다"가 아니라, scaling law 실험 범위에서는 primary driver가 아닐 수 있다는 뜻이다.

#### Non-Embedding Parameters

performance scaling에서 embedding parameter보다 non-embedding parameter count `N`이 더 중요한 scale로 다뤄진다.

이유:

- embedding은 vocabulary와 관련되어 shallow/lookup 성격이 강하다.
- transformer block의 non-embedding parameters가 representation/compute capacity를 더 직접적으로 결정한다.

#### Architecture: Transformer vs LSTM

Transformer는 LSTM보다 long-range context를 잘 활용하고, parameter efficiency 측면에서 더 좋은 scaling behavior를 보인다.

시험 답안:

> 같은 non-embedding parameter 수라도 architecture가 parameter를 얼마나 효율적으로 쓰는지가 다르다. Transformer는 long context와 parallel attention 덕분에 LSTM보다 scale 증가의 이득을 더 잘 활용한다.

#### Transfer Learning

Kaplan 결과에서는 training distribution에서의 improvement가 cross-domain evaluation에도 상당 부분 transfer되는 것으로 소개된다.

의미:

- scale이 커질수록 general language modeling 성능 개선이 다른 distribution에도 이어질 수 있다.
- 단, domain shift와 data quality가 심하면 transfer가 제한될 수 있다.

#### Compute-Optimal Model Behavior

주어진 compute `C`에서 optimal model size가 존재한다. 너무 작은 model은 많이 train해도 capacity가 부족하고, 너무 큰 model은 충분히 train하지 못해 undertrained가 된다.

Kaplan-style conclusion:

- compute가 증가할수록 training duration보다 model size를 늘리는 것이 효율적이라고 주장.

Chinchilla-style correction:

- 기존 큰 모델들이 oversized and undertrained였고, 같은 compute에서는 smaller model을 more tokens로 train하는 것이 더 낫다고 주장.

### Limitations of Power-Law Scaling

power-law scaling은 regime-dependent다.

세 regime:

1. small-data/optimization/finite-size regime
   - scaling이 불안정하다.
   - optimization issue와 finite-size effect가 지배한다.
2. power-law regime
   - log-log plot에서 안정적인 직선 trend가 나타난다.
   - extrapolation이 상대적으로 가능하다.
3. irreducible error regime
   - Bayes error, label noise, intrinsic uncertainty 때문에 더 이상 줄지 않는 floor가 있다.

prep note 질문:

> Which regions are not covered in our scaling law equations?

답안:

- 단순 pure power-law equation은 small-data unstable regime과 irreducible-error plateau를 잘 설명하지 못한다.
- loss floor `E`를 추가하면 irreducible error 일부를 반영할 수 있다.
- small-data regime은 piecewise model, regime indicator, optimization term 등이 필요하다.

### Data Quality

data quality는 단순 data quantity `D`와 다르다.

영향:

- low-quality data는 scaling curve의 offset/intercept를 나쁘게 만든다.
- distribution shift는 data exhaustion cliff를 만들 수 있다.
- duplicate/repeated data는 unique data보다 효과가 작다.

scaling law에 넣는 방법:

- effective data size `D_eff`
- data mixture-dependent offset term
- quality/diversity score를 반영한 weighted token count
- dataset composition variable `q`

### Given Graph Discussion Checklist

scaling law 그래프를 보고 토론하라는 문제가 나오면 다음 순서로 답한다.

1. 축이 linear인지 log-log인지 확인한다.
2. 직선 구간이 있으면 power-law regime과 slope, 즉 scaling exponent를 언급한다.
3. 작은 scale에서 직선이 깨지면 optimization/finite-size/small-data regime이라고 해석한다.
4. 큰 scale에서 plateau가 보이면 irreducible error, data exhaustion, data quality limit를 의심한다.
5. 같은 compute curve에서 loss minimum이 있으면 compute-optimal allocation, 즉 너무 작은 model과 너무 큰 model 사이의 tradeoff를 설명한다.

### Failure Modes

data exhaustion cliff:

- high-quality data가 고갈된 뒤 low-quality/distribution-shifted data를 추가하면 성능이 포화 또는 악화된다.

precision/system cliff:

- FP16/BF16 precision, optimizer instability, system bottleneck 때문에 scale을 키워도 gain이 사라진다.

### Lecture 16 시험형 질문 답안 포인트

- scaling law의 목적: 작은 실험으로 큰 training outcome을 예측하고 compute allocation을 결정한다.
- power-law log-log 해석: log-log plot에서 직선, slope가 scaling exponent.
- non-embedding parameter가 중요한 이유: transformer block capacity와 직접 관련된다.
- simple power-law의 한계: small-data instability, irreducible error floor, data quality, system cliff.

---

## 17. Intermediate Scaling Laws

### Chinchilla

Chinchilla의 핵심 메시지:

> 기존 LLM들은 oversized and undertrained였고, fixed compute budget에서는 model size `N`과 training tokens `D`를 균형 있게 키워야 한다.

Kaplan vs Chinchilla:

| 항목 | Kaplan | Chinchilla |
|---|---|---|
| 결론 | model size scaling이 더 중요 | data scaling도 equally critical |
| token allocation | 상대적으로 적은 token | 더 많은 token |
| 문제점 | fixed LR schedule 이슈 | token 수에 맞춘 cosine LR |
| scaling 비율 | `N:D` growth roughly `0.73:0.27` | roughly `1:1` growth |
| practical message | bigger model 우선 | compute-optimal smaller model + more data |

Chinchilla rule of thumb:

```text
D ~= 20N tokens
```

예: 70B parameter model이면 약 1.4T tokens 수준.

### Kaplan's Analysis with Fixed LR Schedule

Kaplan은 fixed cosine learning rate schedule을 사용했고, 실제 training tokens `D`에 충분히 맞춰 decay되지 않는 경우가 있었다.

문제:

- LR이 충분히 decay되지 않으면 model이 minimum 주변에서 oscillate한다.
- training loss가 필요 이상으로 높아진다.
- data scaling의 이득이 과소평가될 수 있다.

따라서 Kaplan은 `N`의 중요성을 과대평가하고 `D`의 중요성을 과소평가했다는 해석이 가능하다.

### Approach 1: Fix Model Sizes N and Vary D

절차:

1. model size `N`을 고정한다.
2. training token 수 `D`를 다양하게 바꾼다.
3. 각 compute budget에서 가능한 최소 loss envelope를 찾는다.
4. power law를 fit해 compute budget `C`에 대한 optimal `N`과 `D`를 추정한다.

핵심:

- 같은 `N`에서도 더 많이 train하면 loss가 내려간다.
- fixed compute에서 너무 큰 `N`은 충분히 train하지 못해 비효율적이다.

### Approach 2: IsoFLOP Profiles

IsoFLOP은 compute budget을 고정하고 model size를 바꾸는 분석이다.

절차:

1. fixed FLOP budget `C`를 정한다.
2. 여러 `N`을 시도하되, `D`를 조절해 total FLOPs를 같게 만든다.
3. validation loss가 가장 낮은 `N`을 찾는다.
4. 여러 budget에서 optimal point를 이어 efficient frontier를 만든다.

compute approximation:

```text
C ~= 6 N D
```

해석:

- 작은 `N`: capacity 부족, 많이 train해도 한계.
- 큰 `N`: data 부족, undertrained.
- 중간 optimal `N`: compute budget에서 가장 낮은 validation loss.

### Approach 3: Fitting a Parametric Loss Function

Chinchilla는 loss를 `N`과 `D`의 함수로 fit한다.

대표 form:

```text
L(N, D) = E + A / N^alpha + B / D^beta
```

의미:

- `E`: irreducible loss
- `A / N^alpha`: finite model size 때문에 남는 loss
- `B / D^beta`: finite data 때문에 남는 loss

constraint:

```text
C ~= 6 N D
```

optimization 직관:

- `N`을 늘리면 model-size term이 줄어든다.
- `D`를 늘리면 data term이 줄어든다.
- fixed compute에서는 둘 중 하나를 늘리면 다른 하나를 줄여야 한다.

Lagrange/substitution 결과:

```text
N_opt ∝ C^{beta/(alpha + beta)}
D_opt ∝ C^{alpha/(alpha + beta)}
```

따라서 `alpha`와 `beta`가 비슷하면 `N`과 `D`는 compute가 커질 때 비슷한 비율로 증가한다. 이것이 Chinchilla의 `1:1` scaling intuition이다.

### Optimal Solution

optimal frontier에서는 model-size loss term과 data-size loss term의 marginal benefit이 균형을 이룬다.

직관적 답안:

> compute-optimal point는 parameter를 하나 더 늘렸을 때의 loss 감소 이득과 token을 더 학습했을 때의 loss 감소 이득이 compute cost 대비 같아지는 지점이다.

### Data Constraints in Scaling Law

ideal scaling law는 high-quality unique data를 충분히 한 번씩만 사용한다고 가정한다.

현실 문제:

- high-quality data는 finite하다.
- 같은 data를 여러 epoch 반복해야 할 수 있다.
- lower-quality 또는 synthetic data를 섞어야 할 수 있다.

### Scaling Law of Data Repetition

data repetition에서는 repeated token의 가치가 unique token보다 낮아진다.

슬라이드 핵심:

- 반복은 약 4 epoch까지는 가치가 유지될 수 있다.
- 약 40 epoch에 가까워지면 return이 거의 0에 가까워질 수 있다.
- repeated data를 반영하려면 total tokens `D` 대신 effective data `D_eff`가 필요하다.

modified equation 형태:

```text
L(N, D_eff) = E + A / N^alpha + B / D_eff^beta
```

여기서:

- `D_eff <= total repeated tokens`
- repeat count가 커질수록 marginal effective data gain은 감소한다.

결과:

- data-constrained setting에서는 optimal allocation이 달라진다.
- 때로는 더 작은 model을 더 많은 epoch로 train하는 것이 나을 수 있다.

### Scaling Law of Data Composition

dataset mixture는 scaling curve의 slope보다 offset을 바꾸는 것으로 소개된다.

Hashimoto-style idea:

```text
loss scale = size-dependent power law + mixture-dependent offset C(q)
```

의미:

- data source mixture `q`가 좋으면 같은 scale에서도 loss가 낮다.
- small-scale pilot runs로 mixture-dependent term을 추정해 large-scale 성능을 예측할 수 있다.

### Repetition and Composition: Multiple Curves and Optimal Strategy

data curation은 compute-agnostic하지 않다.

small compute budget:

- aggressive filtering이 유리할 수 있다.
- high-quality data만 써도 충분하다.

large compute budget:

- high-quality data만으로는 부족하다.
- lower-quality data를 추가해 quantity를 확보하는 것이 필요할 수 있다.

즉 optimal data mixture는 compute budget에 따라 변한다.

### Hyperparameters in Scaling Law

large-scale hyperparameter tuning은 직접 하기 어렵다. 작은 proxy model에서 튜닝하고 큰 model로 extrapolate하려 하지만, optimal hyperparameter가 scale에 따라 변할 수 있다.

#### Architecture

Transformer는 LSTM보다 scale이 커질 때 더 좋은 asymptotic behavior를 보인다.

답안:

> architecture choice는 scaling curve 자체를 바꾼다. efficient architecture는 같은 compute에서 더 낮은 loss frontier를 만든다.

#### Optimizer

슬라이드 기준:

- Hestness et al.: Adam이 SGD보다 generalization이 좋게 관찰됨.
- Chinchilla: AdamW가 Adam보다 training loss에서 consistently better.
- modern LLM은 stability와 downstream performance 때문에 AdamW를 주로 사용.

#### Aspect Ratio

depth/width aspect ratio는 fixed parameter count에서 loss에 상대적으로 약한 영향을 준다.

답안 주의:

- aspect ratio가 전혀 중요하지 않다는 뜻은 아니다.
- 지나치게 극단적인 architecture나 hardware efficiency는 별도 고려가 필요하다.

#### Batch Size

critical batch size는 compute efficiency와 parallel training speed를 연결한다.

핵심:

- model performance가 좋아질수록 larger batch를 효율적으로 사용할 수 있다.
- critical batch size는 model size보다 loss/performance level에 더 의존할 수 있다.

### Limitations of Scaling Laws

loss-based scaling law가 reasoning, planning 같은 emergent capability를 정확히 예측하는지는 제한적이다.

이유:

- loss는 average token prediction quality를 측정한다.
- reasoning benchmark는 thresholded/nonlinear behavior를 보일 수 있다.
- data quality, diversity, contamination, task distribution이 metric에 큰 영향을 준다.

개선 방향:

- task-specific scaling law
- benchmark performance scaling과 loss scaling의 joint model
- data quality/diversity metric을 `D_eff`나 mixture term에 포함
- small-scale trials로 hyperparameter scaling trend를 fit

### Lecture 17 시험형 질문 답안 포인트

- Chinchilla가 Kaplan을 수정한 이유: fixed LR schedule과 undertraining 문제 때문에 data의 중요성이 과소평가되었다.
- `N:D ~= 1:1` 의미: parameter count와 training tokens가 compute 증가에 대해 비슷한 비율로 커져야 한다는 성장률의 의미다.
- `D ~= 20N`은 practical token-per-parameter rule of thumb이다.
- IsoFLOP profile: fixed compute에서 model size를 바꾸며 validation loss minimum을 찾는 분석.
- parametric loss: `E + A/N^alpha + B/D^beta`, compute constraint `C ~= 6ND`.
- data repetition: repeated tokens는 diminishing return을 가지므로 `D_eff`로 모델링해야 한다.
- data composition: mixture quality는 curve offset을 바꾸며, optimal curation은 compute budget에 따라 달라진다.
- scaling laws의 한계: emergent reasoning/planning, data quality, system cliff, hyperparameter evolution을 단순 loss law만으로 완전히 설명하기 어렵다.

---

## 핵심 공식 모음

### TD Learning

```text
V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]
```

### Q-Learning

```text
Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
```

### Approximate Q-Learning

```text
Q(s,a) = sum_i w_i f_i(s,a)
delta = r + gamma max_a' Q(s',a') - Q(s,a)
w_i <- w_i + alpha delta f_i(s,a)
```

### PPO Ratio and Clipping

```text
r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
objective ~= min(r_t A_t, clip(r_t, 1-epsilon, 1+epsilon) A_t)
```

### Dense Layer Parameter Count

```text
W: output_dim x input_dim
b: output_dim
params = input_dim * output_dim + output_dim
```

### Chain Rule in Backprop

```text
downstream gradient = upstream gradient * local gradient
```

### Power-Law Scaling

```text
L(x) = L_infinity + A x^{-alpha}
log(L - L_infinity) = log A - alpha log x
```

### Chinchilla Parametric Loss

```text
L(N,D) = E + A / N^alpha + B / D^beta
C ~= 6ND
N_opt ∝ C^{beta/(alpha+beta)}
D_opt ∝ C^{alpha/(alpha+beta)}
```

---

## 자주 나올 T/F 함정

- "RL does not assume MDP." False. RL usually assumes MDP structure but does not know `T` and `R`.
- "Direct evaluation uses Bellman backup." False. It averages observed returns; TD uses bootstrapping.
- "Q-learning must follow the optimal policy during training." False. It is off-policy, but must explore enough.
- "epsilon-greedy has low regret because it eventually explores everything." False. It may eventually learn, but random exploration can incur high regret.
- "Approximate Q-learning removes the need for features." False. It depends heavily on feature design/representation.
- "Without activation functions, deeper neural networks are still more expressive." False for pure affine/linear layers; they collapse to one affine map.
- "Training and inference use the same memory." False. Training stores activations and graph for backprop.
- "Sigmoid always improves gradient flow because it is smooth." False. Saturation causes vanishing gradients.
- "Scaling law means performance improves forever." False. Irreducible error, data exhaustion, and system cliffs break simple power laws.
- "Chinchilla says only data matters." False. It says model size and data should be balanced under fixed compute.
- "Loss-based scaling laws automatically predict reasoning ability." False. Reasoning benchmarks may require task-specific models and data-quality terms.

---

## 마지막 회독 체크리스트

- Lecture 10: TD update와 Q-learning update를 구분해서 쓸 수 있는가?
- Lecture 10: 왜 `V(s)`만으로는 model-free control이 어려운가?
- Lecture 11: epsilon-greedy, exploration function, regret를 비교할 수 있는가?
- Lecture 11: approximate Q-learning weight update를 feature 관점에서 설명할 수 있는가?
- Lecture 12: PPO와 GRPO의 차이, DeepVideo-R1의 vanishing advantage 문제를 설명할 수 있는가?
- Lecture 13: CoT, soft prompt, hard prompt, system prompt optimization을 구분할 수 있는가?
- Lecture 13: TextGrad, GEPA, PRESTO가 각각 어떤 feedback/search structure를 쓰는지 설명할 수 있는가?
- Lecture 14: non-linear activation이 필요한 이유와 parameter count를 계산할 수 있는가?
- Lecture 15: upstream/local/downstream gradient와 vanishing/exploding gradient 원인을 설명할 수 있는가?
- Lecture 16: power law가 log-log plot에서 직선이 되는 이유를 쓸 수 있는가?
- Lecture 16: simple scaling law가 커버하지 못하는 regime을 말할 수 있는가?
- Lecture 17: Kaplan vs Chinchilla, IsoFLOP, parametric loss derivation intuition을 설명할 수 있는가?
- Lecture 17: data repetition/composition이 `D`를 단순 token count로 볼 수 없게 만드는 이유를 설명할 수 있는가?
