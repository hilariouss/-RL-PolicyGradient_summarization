# -RL-PolicyGradient_summarization
1. 이 저장소는 **강화학습(RL, Reinforcement learning)** 의 학습 방법 중 하나인 **정책 경사(***policy gradient***)** 에 대한 방법론을 정리합니다.
원문은 다음 링크 >https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html 을 참고하였음을 밝힙니다.

2. 또한 본 저장소는 **policy gradient**에 대한 전반적인 내용을 정리함과 더불어, 다양한 **policy gradient 기반의 심층 강화학습 알고리즘들을 직접 구현하고, 해당 소스코드를 공유하는 것을 목적으로 만들어졌습니다.**
저의 논문 작성과 연구를 위한 작은 저장소가 많은 사람들에게 도움이 되었으면 합니다 :)

# Preliminary. Value-based RL vs. Policy-based RL
고전적 강화학습 알고리즘들은 **1)어떤 상태에서 취할 수 있는 행동별 가치를 나타내는 action-value function인 Q-function을 approximate/estimate하고, 2)해당 Q-function을 활용해 특정 (상태, 행동) pair에 대한 값들을 계산, *ε*-greedy와 같은 방법으로 agent가 최적의 행동을 수행할 수 있도록 하였습니다.** 이러한 방법은 value-based 강화학습 이라고 할 수 있습니다.
즉, value-bawed RL은 **Q(s, a)의 value** 를 계산, 비교함으로써, agent가 일련의 의사결정(sequential decision making)을 수행할 수 있도록 하는 것을 기본 아이디어로 삼습니다. 이를 위해서 최적의 Q-function을 계산하고, 그 과정에서 Bellman optimality equation 을 활용합니다.

본격적으로 정책경사에 대한 설명을 시작하기 앞서, 강화학습에 대한 기본적인 사항이 숙지가 필요하다면 https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts 링크를 먼저 참고하시면 좋습니다.

# 1. Preliminary of policy gradient
Policy gradient는 강화학습 문제들을 푸는 방법입니다. **Policy gradient는 policy로 표현되는 목적함수(보상함수)에 대한 gradient를 계산하고, 이를 활용해 expected future return을 최대화 하도록 policy를 조절하는 방법입니다.** Policy gradient에 대한 증명 및 자세한 사항은 Richard S. Sutton 교수님의 policy gradient 논문 > http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf 을 참고하시길 바랍니다. 본격적으로 policy gradient를 설명하기에 앞서, 앞으로 사용할 notation에 대한 정리를 하도록 하겠습니다.

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
| *A*(*s*, *a*) | Advantage function. *A*(*s*, *a*) = *Q*(*s*, *a*) - *V*(*s*). *V*(*s*)를 baseline으로 하여, 상태 *s*에서 취할 수 있는 행동 *a*별 우수성을 나타내는데 사용하는 함수입니다. |

# 2. Proof of policy gradient

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

