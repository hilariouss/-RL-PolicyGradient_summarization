# -RL-PolicyGradient_summarization
1. 이 저장소는 **강화학습(RL, Reinforcement learning)** 의 학습 방법 중 하나인 **정책 경사(***policy gradient***)** 에 대한 방법론을 정리합니다.
원문은 다음 링크 >https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html 을 참고하였음을 밝힙니다.

2. 또한 본 저장소는 **policy gradient**에 대한 전반적인 내용을 정리함과 더불어, 다양한 **policy gradient 기반의 심층 강화학습 알고리즘들을 직접 구현하고, 해당 소스코드를 공유하는 것을 목적으로 만들어졌습니다.**
저의 논문 작성과 연구를 위한 작은 저장소가 많은 사람들에게 도움이 되었으면 합니다 :)

# 0. Value-based RL vs. Policy-based RL
고전적 강화학습 알고리즘들은 **1)어떤 상태에서 취할 수 있는 행동별 가치를 나타내는 action-value function인 Q-function을 approximate/estimate하고, 2)해당 Q-function을 활용해 특정 (상태, 행동) pair에 대한 값들을 계산, ε-greedy와 같은 방법으로 agent가 최적의 행동을 수행할 수 있도록 하였습니다.** 이러한 방법은 value-based 강화학습 이라고 할 수 있습니다.
즉, value-bawed RL은 **Q(s, a)의 value** 를 계산, 비교함으로써, agent가 일련의 의사결정(sequential decision making)을 수행할 수 있도록 하는 것을 기본 아이디어로 삼으며, 이를 위해 최적의 Q-function을 계산하기 위해 Bellman optimality equation 을 활용하여 계산합니다.

본격적으로 정책경사에 대한 설명을 시작하기 앞서, 강화학습에 대한 기본적인 사항이 숙지가 필요하다면 https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts 링크를 먼저 참고하시면 좋습니다.

# 1. Preliminary of policy gradient
Policy gradient는 강화학습 문제들을 푸는 방법입니다. **Policy gradient는 policy로 표현되는 목적함수에 대한 gradient를 계산하고, 이를 활용해 expected future return을 최대화 하도록 policy를 조절하는 방법입니다.** Policy gradient에 대한 증명 및 자세한 사항은 Richard S. Sutton 교수님의 policy gradient 논문 > http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf 을 참고하시길 바랍니다. 본격적으로 policy gradient를 설명하기에 앞서, 앞으로 사용할 notation에 대한 정리를 하도록 하겠습니다.

|  Symbol | Description |
|:-------:|:-----------: |
| s ∈ *S* | Agent의 상태.|
| a ∈ *A* | Agent의 행동.|
| r ∈ *R* | Environment로부터의 보상.|
| s<sub>t</sub>, a<sub>t</sub> , r<sub>t</sub> | 어떤 trajectory 에서 time step *t* 일 때 agent의 상태, 행동, 보상.|
| γ | 현재 보상 대비 미래 기대보상에 대한 페널티, 감쇠상수. (0 < γ ≤ 1)| 
| G<sub>t</sub> | 누적 기대 보상값. Agent가 학습하는 척도로, 이를 최대화 하도록 자신의 행동양식(정책)을 최적화합니다. ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_gif/G_10.png)|
| *P*(*s*<sup>'</sup>, *r* \| *s*, *a*) | 상태 전이 확률(state transition probability). Agent가 현재 상태와 선택한 행동이 각각 *s*와 *a*일 때, 다음 상태가 *s*<sup>'</sup>이고, 보상으로 *r*을 받을 상태 전이 확률.|
| a ∈ *A* | a: 행동, *A*: 행동 집합 |
| a ∈ *A* | a: 행동, *A*: 행동 집합 |
| a ∈ *A* | a: 행동, *A*: 행동 집합 |

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

