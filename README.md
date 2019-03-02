# -RL-PolicyGradient_summarization
1. 이 저장소는 **강화학습(RL, Reinforcement learning)** 의 학습 방법 중 하나인 **정책 경사(***policy gradient***)** 에 대한 방법론을 정리합니다.
원문은 [이곳](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)을 참고하였음을 밝힙니다.

2. 또한 본 저장소는 **policy gradient**에 대한 전반적인 내용을 정리함과 더불어, 다양한 **policy gradient 기반의 심층 강화학습 알고리즘들을 직접 구현하고, 해당 소스코드를 공유하는 것을 목적으로 만들어졌습니다.** 

혹시 강화학습에 대한 기본적인 사항이 숙지가 필요하다면 [이곳](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts)을 먼저 참고하시면 좋습니다.
저의 논문 작성과 연구를 위한 이 작은 저장소가 많은 사람들에게 도움이 되었으면 합니다 :)

<hr>

# 0. Preliminary
Policy gradient는 강화학습 문제들을 푸는 방법입니다. **Policy gradient는 policy로 표현되는 목적함수(보상함수)에 대한 gradient를 계산하고, 이를 maximize하는 행동에 대한 확률을 증가시켜 expected future return을 최대화 하도록 policy를 조절하는 방법입니다.** Policy gradient에 대한 증명 및 자세한 사항은 Richard S. Sutton 교수님의 policy gradient [논문](http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)을 참고하시길 바랍니다. 본격적으로 policy gradient를 설명하기에 앞서, 앞으로 사용할 notation에 대한 정리를 하도록 하겠습니다.

|  Symbol | Description |
|:-------:|:-----------: |
| *s* ∈ *S* | Agent의 **상태**.|
| *a* ∈ *A* | Agent의 **행동**.|
| *r* ∈ *R* | Environment로부터의 **보상**.|
| *s*<sub>*t*</sub>, *a*<sub>*t*</sub> , *r*<sub>*t*</sub> | 어떤 trajectory 에서 time step *t* 일 때 agent의 상태, 행동, 보상.|
| *γ* | 현재 보상 대비 미래 보상에 대한 페널티, 또는 감쇠상수. (0 < *γ* ≤ 1)| 
| *G*<sub>*t*</sub> | Return. Agent 의 학습척도. ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_gif/G_10.png) (**return은 누적 기대 보상값과 동일**합니다.) |
| *P* (*s*<sup>'</sup>,*r*\|*s*,*a*) | 상태 전이 확률(state transition probability).|
| π<sub>θ</sub> (*a*\|*s*) | Stochastic policy. 정책 π를 기준으로 *s*일 때 *a*를 행할 확률. π는 policy parameter θ로 표현됩니다.|
| µ(*s*) | Deterministic policy. π와 명시적으로 구별하기 위한 다른 표기를 사용합니다. |
| *V*(*s*) | State-value function. Agent의 상태에 대한 미래 가치를 나타냅니다. 특정 정책을 따르는 것에 구애받지 않고 가치 기반 강화학습을 할 경우 활용하며, *V*(*s*)가 ω로 parameterized될 경우 V<sub>ω</sub> (*s*)와 같이 표기될 수 있습니다. *ϵ*-greedy와 같은 학습 방법을 사용 예로 들 수 있습니다. |
| *V*<sup>*π*</sup>(*s*) | Policy π를 따르는 가치함수 *V*<sup>*π*</sup>(.)로 상태 *s*의 (expected) return을 나타냅니다; ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_gif/V_pi.png) |
| *Q*(*s*, *a*) | Action-value function. *V*(*s*)가 **상태**의 가치를 나타냈다면, *Q*(*s*, *a*)는 행동별 return을 나타냅니다. 역시 ω로 parameterized될 경우 Q<sub>ω</sub> (*s*, *a*)와 같이 표기될 수 있습니다. |
| *Q*<sup>π</sup>(*s*, *a*) | *V*<sup>*π*</sup>(*s*)와 유사하게 **상태와 행동 pair (*s*,*a*)** 에 대해 policy π를 따르는 action-value function (Q-function)의 값(return)을 나타냅니다; ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_gif/Q_pi.png) |
| *A*(*s*, *a*) | Advantage function. *V*(*s*)를 baseline으로 하여, 상태 *s*에서 취할 수 있는 행동 *a*별 우수성을 나타내는데 사용하는 함수입니다. ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_gif/A_10.png)|

<hr>

# 1. Value-based reinforcement learning
Policy gradient 방식의 강화학습을 살펴보기 앞서, 비교 대상인 value-based 강화학습 기법을 간단하게 살펴보겠습니다. Value-based RL에 대한 개요 및 다양한 알고리즘의 설명이 궁금하시면 [이곳](https://github.com/hilariouss/-RL-Value_based_summariaztion)을 참고해주시기 바랍니다. :)

Value-based reinforcement learning은 **상태와 행동에 대한 미래 가치를 나타내는 value function의 값을 계산하고 이 값을 활용하여 agent가 최적의 행동을 할 수 있도록 설계되었습니다.** 이는 value function이 **상태에 대한 미래 가치를 나타내는 state-value function** 이거나, **상태와 그 상태에서 행할 수 있는 행동들 별 가치인 action value function (Q-function)** 로 양분할 수 있습니다: 

<p style="background-color: #171515">
  
**1) state-value function인 경우, 특정 상황의 가치를 iteration 과정을 반복해 모든 상황들에 대해 미래 가치를 업데이트 해나갑니다. 갱신되는 상태별 미래 가치 값에 따라 agent는 *ε*-greedy 방식과 같은 옵션을 선택해 행동할 수 있습니다.**

**2) action-value function인 경우, 특정 상황에서의 각 행동에 대한 미래 가치를 iteration 과정을 반복해 어떤 상황에 agent가 도달했을 때, 미래 가치에 대한 기댓값이 가장 높은 행동을 취하도록 행동할 수 있습니다.**

</p>

하지만, 이러한 상태별, 또는 상태 및 행동별 가치를 도출하기 위한 state-value function 또는 action-value function의 design 및 실 적용을 위한 state/action space의 continuity 및 dimension 문제로 인해 최근 deep neural network를 활용한 연구가 활발히 이루어져 Deep Q-Network(DQN)과 같은 기법이 등장했습니다. 이러한 deep learning 기반의 value-based 강화학습은 가치 함수의 approximation(근사)를 통한 유도에 있다고 할 수 있겠습니다. 정리하면 이러한 value-based 학습 방법은 **최적 가치 함수에 대한 근사 및 Q 값의 비교를 통한 agent의 행동 '유도'에 초점**이 맞추어져 있습니다. 우리가 앞으로 살펴볼 policy gradient 방식의 학습은 Q 함수와 같은 가치함수의 근사과정을 생략하고 직접 **최적 정책에 대한 학습**을 수행한다는 측면에서 구별된다고 할 수 있습니다.

<hr>

# 2. Introduction and goal of *Policy gradient*  

# 3. Policy gradient 알고리즘
## 3-1. REINFORCE (xxxx)
## 3-2. Actor-critic (Vanilla policy gradient, xxxx)
## 3-3. Off-policy policy gradient (xxxx)

## 3-4. A2C (Advantage Actor-Critic)
## 3-5. A3C (Asynchronous Advantage Actor-Critic)
## 3-6, DPG (Deterministic Policy Gradient)
## 3-7. DDPG (Deep Deterministic Policy Gradient)
## 3-8. D4PG (Distributed Distributional DDPG)
## 3-9. MADDPG (Multi-agent DDPG)
## 3-10. TRPO (Trust Region Policy Optimization)
## 3-11. PPO (Proximal policy optimization)
## 3-12. ACER (Actor-Critic with Experience Replay)
## 3-13. ACKTR (Actor-Critic using Kronecker-factored Trust Region)
## 3-14. SAC (Soft Actor-Critic)
## 3-15. TD3 (Twin Delayed Deep Deterministic)

# References

*Todo (0/17)*
- [ ] 1. Preliminary of policy gradient
- [ ] 2. Proof of policy gradient
- [ ] 3. Policy gradient algorithm
  - [ ] 3-1. REINFORCE
  - [ ] 3-2. Vanilla policy gradient
  - [ ] 3-3. Off-policy policy gradient
  - [ ] 3-4. A2C
  - [ ] 3-5. A3C
  - [ ] 3-6. DPG
  - [ ] 3-7. DDPG
  - [ ] 3-8. D4PG
  - [ ] 3-9. MADDPG
  - [ ] 3-10. TRPO
  - [ ] 3-11. PPO
  - [ ] 3-12. ACER
  - [ ] 3-13. ACKTR
  - [ ] 3-14. SAC
  - [ ] 3-15. TD3

