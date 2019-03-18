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
| *G*<sub>*t*</sub> | Return. Agent 의 학습척도. ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/0.Preliminary/G_10.png) (**return은 누적 기대 보상값과 동일**합니다.) |
| *P* (*s*<sup>'</sup>,*r*\|*s*,*a*) | 상태 전이 확률(state transition probability).|
| π<sub>θ</sub> (*a*\|*s*) | Stochastic policy. 정책 π를 기준으로 *s*일 때 *a*를 행할 확률. π는 policy parameter θ로 표현됩니다.|
| µ(*s*) | Deterministic policy. π와 명시적으로 구별하기 위한 다른 표기를 사용합니다. |
| *V*(*s*) | State-value function. Agent의 상태에 대한 미래 가치를 나타냅니다. 특정 정책을 따르는 것에 구애받지 않고 가치 기반 강화학습을 할 경우 활용하며, *V*(*s*)가 ω로 parameterized될 경우 V<sub>ω</sub> (*s*)와 같이 표기될 수 있습니다. *ϵ*-greedy와 같은 학습 방법을 사용 예로 들 수 있습니다. |
| *V*<sup>*π*</sup>(*s*) | Policy π를 따르는 가치함수 *V*<sup>*π*</sup>(.)로 상태 *s*의 (expected) return을 나타냅니다; ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/0.Preliminary/V_pi.png) |
| *Q*(*s*, *a*) | Action-value function. *V*(*s*)가 **상태**의 가치를 나타냈다면, *Q*(*s*, *a*)는 행동별 return을 나타냅니다. 역시 ω로 parameterized될 경우 Q<sub>ω</sub> (*s*, *a*)와 같이 표기될 수 있습니다. 본문에서 소문자 q를 혼용할 수 있습니다. |
| *Q*<sup>π</sup>(*s*, *a*) | *V*<sup>*π*</sup>(*s*)와 유사하게 **상태와 행동 pair (*s*,*a*)** 에 대해 policy π를 따르는 action-value function (Q-function)의 값(return)을 나타냅니다; ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/0.Preliminary/Q_pi.png) |
| *A*(*s*, *a*) | Advantage function. *V*(*s*)를 baseline으로 하여, 상태 *s*에서 취할 수 있는 행동 *a*별 우수성을 나타내는데 사용하는 함수입니다; ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/0.Preliminary/A_10.png)|

<hr>

# 1. Value-based reinforcement learning
Policy gradient 방식의 강화학습을 살펴보기 앞서, 비교 대상인 value-based 강화학습 기법을 간단하게 살펴보겠습니다. Value-based RL에 대한 개요 및 다양한 알고리즘의 설명이 궁금하시면 저의 저장소 [이곳](https://github.com/hilariouss/-RL-Value_based_summariaztion)을 참고해주시기 바랍니다. :)

Value-based reinforcement learning은 **상태와 행동에 대한 미래 가치를 나타내는 value function의 값을 계산하고 이 값을 활용하여 agent가 최적의 행동을 할 수 있도록 설계되었습니다.** 이는 value function이 **상태에 대한 미래 가치를 나타내는 state-value function** 이거나, **상태와 그 상태에서 행할 수 있는 행동들 별 가치인 action value function (Q-function)** 로 양분할 수 있습니다: 

```
(1) state-value function인 경우, 특정 상황의 가치를 iteration 과정을 반복해 모든 상황들에 대해 미래 가치를 업데이트 
해나갑니다. 갱신되는 상태별 미래 가치 값에 따라 agent는 ε-greedy 방식과 같은 옵션을 선택해 행동할 수 있습니다.

(2) action-value function인 경우, 특정 상황에서의 각 행동에 대한 미래 가치를 iteration 과정을 반복해 어떤 상황에 
agent가 도달했을 때, 미래 가치에 대한 기댓값이 가장 높은 행동을 취하도록 행동할 수 있습니다. 
```

하지만, 이러한 상태별, 또는 상태 및 행동별 가치를 도출하기 위한 state-value function 또는 action-value function기반의 방식은 real world에 적용하였을 때 발생하는 문제(massive state space dimension)로 인해 최근 deep neural network를 활용한 연구가 활발히 이루어져 Deep Q-Network(DQN)과 같은 기법이 등장했습니다. 이러한 deep learning 기반의 value-based 강화학습은 가치 함수의 approximation(근사)를 통한 유도에 있다고 할 수 있겠습니다. 

```
정리하면 이러한 value-based 학습 방법은 최적 가치 함수에 대한 근사 및 Q 값의 비교를 통한 agent의 행동 '유도'에 초점이 맞추어져 있습니다. 
우리가 앞으로 살펴볼 policy gradient 방식의 학습은 Q 함수와 같은 가치함수의 근사과정을 생략하고 직접 최적 정책에 대한 학습을 수행한다는 측면에서 구별된다고 할 수 있습니다. 
이를통해 추가적인 Q 값의 비교 작업 없이도 학습한 최적 정책에 상태를 입력하면 최적의 행동이 어떤 것인지 즉각적인 의사결정이 가능합니다.
```
<hr>

# 2. Introduction and goal of *Policy gradient* 
**Policy gradient**는 policy 자체를 직접적으로 modeling하고 optimize하는 방법입니다. Policy는 위 notation과 같이 policy parameter θ로 표현되며, 이는 π<sub>θ</sub> (*a*\|*s*) 라고 했습니다. Policy gradient는 이러한 policy를 포함하는 목적함수를 policy에 대한 parameter θ에 대한 기울기(gradient)를 구하여 목적함수를 최적화 합니다. 이 목적함수는 policy를 포함하는 reward function이 됩니다. 즉, policy를 포함하는 reward 함수를 목적함수로 설정하고, 이에 대한 θ의 gradient를 구해 이 목적함수를 최적화 하여 최대한의 보상을 얻도록 θ를 학습하는 것이 policy gradient라고 할 수 있겠습니다. 그럼 목적 함수(보상함수)를 살펴보겠습니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/J_theta.png)

여기서 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/d^pi(s).png)는 Markov chain의 stationary distribution입니다. 

**Stationary distribution ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/d^pi(s).png)은 state 변화의 sequence를 표현하는 finite Markov chain이 있을 때, 충분히 큰 상태전이를 반복하다보면 어떤 상태에 도달하는 확률들이 수렴한다는 확률분포를 말합니다.** Stationary distribution의 매력적인 점은 초기 Markov chain의 상태전이 확률에 영향을 받지 않는다는 점입니다. 왜냐하면 최종적으로 이 stationary distribution이 수렴하는 확률분포에 도달할 것이기 때문입니다. Markov chain의 상태들에 대해 영원히 transition 하면 결국 어떤 terminal state에 도달한다는 확률이 불변한다는 것이라고 생각하면 좀 더 쉽습니다. 즉, 처음 상태가 *s*<sub>0</sub>이고, policy ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/pi_theta.png)를 따를 때, *t* time step이 흘렀을 때의 상태 *s*<sub>t</sub>가 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/s_{t}=s.png)가 될 확률이 곧 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/d^pi(s).png)가 되는 것입니다.
결국 Stationary distribution을 수식으로 표현하면 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/d^pi_lim.png)가 됩니다.

Policy-based 방식은 continuous space의 state 또는 action space에 대해 학습하는데 더욱 효과적 입니다. Discrete한 state 및 action들에 대한 가치는 value-based 방식에서 사용할 수 있었지만, continuous하여 무한한 state 또는 action space에 대한 문제일 경우 한계가 있습니다. 예를 들어 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/argmax.png)를 계산해 policy를 improve하는 policy iteration의 경우 무한한 action space에 대한 계산이 거의 불가능합니다.

*Gradient ascent* 방식을 사용하여, 우리는 gradient ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/gradient_J.png)에 대해 가장 높은 return을 주는 방향으로 policy를 나타내는 parameter θ를 조절(학습)합니다. Gradient는 아래와 같이 표현됩니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/J(theta)_derivation.png)
이 때, ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/E_pi.png)는 state와 action distribution이 policy ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/pi_theta.png)를 모두 따르는 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/E_sdapi.png)임을 나타냅니다.

결국, (Vanilla) policy gradient는 아래 기댓값으로 표현된 θ에 대한 gradient를 활용합니다. 
![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/J(theta)_derivation_2.png)

하지만 vanilla policy gradient는 bias가 없고, variance가 높아 bias는 유지하면서 variance는 줄이려는 다양한 policy gradient 알고리즘들이 제시되었습니다. 결국, Gradient에 대한 다양한 수식들이 존재하고, 이에 대한 일반식이 [Schulman et al., 2016](https://arxiv.org/abs/1506.02438)에 제시되었습니다. 일반화된 gradient에 대한 계산식 GAE(General advantage estimation)은 아래와 같습니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/2.Intro_and_goal_of_RL/General_Gradient.png)

결국, Policy gradient 기반의 알고리즘들은 위와 같은 gradient를 활용해 expected future return을 maximize시키도록 policy parameter θ를 학습하는 것이라고 할 수 있습니다.

# 3. Policy gradient 알고리즘
## 3-1. REINFORCE ([논문](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)|[코드](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/1.%20REINFORCE/REINFORCE_CartPole.py))
**Monte-Carlo policy gradient**, 또는 **REINFORCE**(R.J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine learning*, vol. 8, pp. 3-4, 1992)는 episode의 샘플들을 활용해 policy parameter θ를 update합니다. 즉 episode에서 estimate할 수 있는 return 값을 활용해 policy parameter를 update 합니다. REINFORCE는 gradient에 대한 actual 값과 expectation of sample gradient가 동일하기 때문에 동작합니다. 무슨 의미인지 이해가 가지 않을 수 있어 아래 수식을 다시 첨부합니다:

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/3-1.REINFORCE/REINFORCE.png)

파란색의 박스는 gradient에 대한 actual value (proportional)를 나타내며, 빨간색의 박스는 gradient에 대한 expectation을 나타냅니다. 이 둘의 값이 동일하기 때문에 REINFORCE는 episode의 sample 값에 대한 return을 계산하고 이를 바탕으로 gradient를 update할 수 있습니다.

REINFORCE가 Monte-Carlo policy gradient라고 불리는 이유는 Monte-Carlo 방법으로 full trajectory(episode)를 구하고, 이를 이루는 샘플들을 바탕으로 return을 계산하여 policy update에 활용하기 때문입니다. REINFORCE의 알고리즘은 아래와 같습니다.

### *REINFORCE algorithm*
![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/raw/master/Equation_img/3-1.REINFORCE/REINFORCE_algo.png)

간단하게 알고리즘을 살펴보면 초기 policy parameter를 랜덤하게 초기화하고, 이를 바탕으로 하나의 trajectory를 생성하는 것을 알 수 있습니다. 이후, 생성한 trajectory의 sample들에 대한 return을 계산하고, iterative하게 policy를 gradient ascent방식으로 갱신하는 것을 확인할 수 있습니다.

한편, REINFORCE의 다른 변화한 버전도 존재하는데, 알고리즘의 가장 아래 return으로 표기된 *G*<sub>t</sub>에서 baseline 역할을 하는 state-value function을 뺀 것을 활용하기도 합니다. 이는 gradient estimation의 variance는 감소시키면서 동시에 bias는 유지하기 위함입니다. 즉, Q 함수로 나타나는 return에서 state-value function을 뺀 advantage 함수 A(s, a)가 *G*<sub>t</sub>를 대체 수 있습니다.

## 3-2. Actor-critic ([논문](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/2.%20Actor-critic/Actor-critic%20algorithm.pdf)|[코드](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/2.%20Actor-critic/Actor_critic.py)|[참고](https://medium.freecodecamp.org/an-intro-to-advantage-actor-critic-methods-lets-play-sonic-the-hedgehog-86d6240171d))
이전에 살펴본 REINFORCE의 Monte-Carlo method는 에피소드가 끝날때 까지 기다렸다가 업데이트를 하는 방식을 채택했습니다. Gradient의 업데이트 수식을 살펴보면, 정책으로 도출되는 행동들에 대한 확률 분포 및 expected future return (*G*<sub>t</sub>)이 있었는데, Monte-Carlo 방식은 이들의 variance가 커 gradient를 급격하게 변화시키고 따라서 안정적인 학습을 수행하는데 한계가 있었습니다. 이에 대한 대안으로 등장한 것이 바로 Actor-critic 알고리즘 입니다.

Actor-critic 알고리즘을 자세히 살펴보기 전, 비교를 통해 이해를 돕고자 합니다. REINFORCE의 Monte-Carlo 방식과 상반되는 방식으로는 Temporal difference(TD) 방식이 있습니다. TD 방식은 다음 time step과의 오차인 TD-error를 이용해 에피소드의 전체 time-step이 다 지날때 까지 업데이트를 미루는 것이 아니라, 현재 time step의 예측 가치와 다음 time-step의 target 가치의 오차를 계산해 업데이트에 활용하는 방식입니다. 이를 위해서는 time step마다 agent가 행동해보고, 그 행동에 대한 가치를 평가하여 agent의 정책을 계속해서 변화해 나갑니다. 이 때 agent의 action을 결정하는 것을 actor라고 합니다. 

한편, actor의 행동을 평가하는 다른 요인이 존재하는데, 이를 critic이라고 합니다. Critic은 actor의 행동으로 일어나는 상태 전이에 대해 value function를 이용, TD-error를 계산해 actor의 policy update가 일어날 수 있도록 합니다. 또한 critic은 상태함수를 TD-error를 활용해 update합니다.
즉, Actor-critic은 actor와 critic이 매 time-step에서 TD-error를 활용해 각각 policy network (parameterized with θ)와 value-function (parameterized with ω)을 업데이트하는 policy gradient 알고리즘이라고 할 수 있습니다. Actor-critic 알고리즘의 구조는 아래 그림과 같습니다. 

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-2.AC/Actor-critic.png)

직관적으로 위 그림을 이해하자면 actor와 critic이 TD-error를 활용해 자신들의 업데이트를 수행한다는 것을 확인할 수 있습니다. Actor가 취한 행동으로 다음 상태와 보상을 환경으로부터 받으면, critic의 value function이 TD-error를 계산하고 이를 actor의 policy network(policy estimator)와 critic의 value function(value estimator)를 업데이트 합니다. 이 때 유의할 점은 actor와 critic은 각자 다른 독립적인 parameter를 업데이트 한다는 점입니다. 

수식으로 REINFORCE의 Monte-Carlo method 기반의 policy update와 actor-critic의 TD-error 기반의 policy update 방식을 살펴보면 아래 그림과 같습니다:

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-2.AC/actor-critic-newupdate-policy.png)

```
기존 REINFORCE의 경우, 마지막 R로 표기된 cumulative future reward (return)이 곱해지기 위해 episode의 
종료까지의 과정이 필요했습니다. 하지만, actor-critic은 이를 대신해 critic의 value function을 활용합니다. 
각 approximator(policy와 value function)의 업데이트는 actor와 critic이 독립적으로 수행하지만, 결국 
이 구조를 통해 critic의 평가가 actor의 policy update에 반영되는 것입니다.
```
Actor와 critic의 update 수식은 아래와 같습니다.
```
1. Actor policy update
```
![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-2.AC/actor_update.png)
```
2. Critic value approximator update
```
![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-2.AC/critic_update.png)

Value function의 update 수식에서 β는 positive step-size parameter로 learning rate역할을 수행하며, policy update의 learning rate α와 구별됩니다. Actor-critic 알고리즘은 아래와 같습니다.

### *(On-policy) Actor-critic algorithm*

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-2.AC/actor_critic_algo.png)

## 3-3. Off-policy policy gradient ([논문1](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/3.%20Off-policy%20policy%20gradient/Off-policy_Actor_critic%20(Off-PAC).pdf)|[논문2](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/3.%20Off-policy%20policy%20gradient/Policy%20Gradient%20Methods%20for%20Off-policy%20control.pdf)|[코드(TBD)](TBD))

위에서 살펴본 두 알고리즘 (REINFORCE, vanlilla actor-critic) 알고리즘은 모두 on-policy 알고리즘으로, sample을 학습하는 policy와 sample을 모으는 policy가 동일한 알고리즘입니다. 즉, 우리가 update하고자 하는 대상인 *target policy*가 곧 sample을 수집하는 *behaviour policy*와 동일합니다. 이러한 on-policy 알고리즘은 학습하는 policy와 sample을 수집하는 policy가 동일하기 때문에, 한 번 업데이트를 하고나면 이전의 experience는 모두 무의미해져 사용할 수 없게 됩니다. 이는 sample efficiency를 저하시키는 요소이며, exploration 및 학습이후 재사용이 불가능하다는 단점이 존재합니다.

하지만, **off-policy 알고리즘은 *target policy*와 *behaviour policy*가 달라, *target policy*의 학습을 위해 과거에 다른 policy로 수집했던 experience도 학습에 사용할 수 있다는 장점**이 있습니다. 즉, 과거의 experience를 target policy update에 활용할 수 있어 on-policy 대비 **sample efficiency가 향상**되었다라고 할 수 있습니다. 

정리하면 *off-policy* 알고리즘에서 *target policy*는 학습의 반영이 이루어지는 policy이며, *behaviour policy*는 exploration을 통한 behavior 시행 및 sample을 수집하는 policy라고 할 수 있습니다. 

이러한 *Off-policy* 알고리즘의 특징은 아래와 같이 두 가지로 정리할 수 있습니다.

```
1. 학습을 위해 full trajectories를 필요로 하지 않으며, 과거 episode를 재사용하는 experience replay(경험 재생)을 통해
sample efficiency를 향상시킵니다.
2. Sample collection들이 target policy를 통해 behaviour policy를 따라서 generation되기 때문에, 더 나은 exploration을
수행할 수 있습니다.
```

그렇다면 어떻게 *off-policy* 알고리즘의 gradient를 계산할 수 있는지 알아보겠습니다. 이전에 짚어본 *on-policy*와의 차이점을 상기해보면, *off-policy*알고리즘은 target policy와 behaviour policy가 따로 존재한다고 했습니다. 즉, 각 policy를 표현하는 parameter가 독립적으로 존재합니다. Target policy를 parameter θ로 parameterize한 것을 π<sub>θ</sub>(*a*|*s*), behaviour policy를 β(*a*|*s*)라고 표현하면 policy gradient의 objective function은 아래와 같이 behaviour policy β(*a*|*s*)로 정의된 상태분포에 대한 보상의 합으로 나타낼 수 있습니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-3.Off-policy_PG/offpolicy_pg_obj_fn.png)

위의 objective function에서 *d*<sup>β</sup>(*s*)는 behaviour policy β의 stationary distribution입니다. 즉, ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-3.Off-policy_PG/offpolicy_pg_state_distribution.png)입니다. 유의할 점은 *Q*<sup>π</sup>는 target policy로 계산된다는 것입니다. Training observation이 행동 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-3.Off-policy_PG/offpolicy_pg_action.png)으로 sampling 된다고 할 때, 위 목적식에 대한 gradient 계산은 아래와 같습니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-3.Off-policy_PG/offpolicy_pg_gradient.png)

여기서 파란색으로 표시된 ![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-3.Off-policy_PG/offpolicy_pg_rho.png)는 target policy의 결과와 behaviour policy의 결과의 비율입니다 (ratio of the target policy to the behaviour policy). 즉, 두 policy간의 비율을 적용한 점과, stationary distribution이 behaviour policy를 따른다는 점이 이전에 살펴본 on-policy policy gradient 알고리즘과의 차이점이라고 할 수 있습니다.

## 3-4. A3C (Asynchronous Advantage Actor-Critic)([논문](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/5.%20A3C/A3C.pdf)|[코드](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/5.%20A3C/A3C.py))
비동기적 어드밴티지 actor-critic(A3C)은 parallel training에 초점을 둔 policy gradient 알고리즘 입니다. **전역 actor-critic 망과 다수의 worker들이 존재하며, 다수의 worker(=agent)들의 gradient 계산 결과를 각자 계산하여 전역망의 actor 및 critic의 gradient를 비동기적으로 업데이트 합니다.**

일반적으로 paper에서 update를 위해 actor와 critic의 loss 계산을 위해서는 TD 에러를 활용하는데, advantage의 approximation으로 loss 값을 계산하는데 활용하고 있습니다. 세부적인 차이에 대한 이해를 위해 아래 정보를 추가합니다.
TD 에러와 advantage, Bellman error의 차이는 아래 그림과 같습니다. ([참조](http://www.boris-belousov.net/2017/08/10/td-advantage-bellman/))

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-4.A2C/td_error_advantage_BE.png)

즉, worker들은 각자 독립적으로 trajectory를 생성하며 전이에 대한 경험을 축적하고, 전역망 갱신 조건(done or update주기)에 도달할경우 TD 에러를 계산, loss에 대한 gradient를 계산합니다. **각 worker들은 각자의 환경에서 독립적으로 환경과 상호작용하기 때문에, 종료 조건에 동시에 도달하지 않을 수 있어 자연스럽게 loss에 대한 gradient를 계산하는 시점도 다를 수 있습니다. 따라서, global network에 업데이트 하는 시점도 제각각일 수 있습니다. 이러한 점이 바로 비동기적 학습인 A3C의 특징이라고 할 수 있습니다.** 아래는 A3C의 알고리즘입니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-5.A3C/a3c_algorithm.png)

## 3-5. A2C (Advantage Actor-Critic)([논문](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/4.%20A2C/A3C.pdf)|[코드](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/4.%20A2C/A2C_CartPole-v0.py))
A2C는 A3C의 *synchronous*하고 deterministic한 버전의 알고리즘입니다. A3C에서 각 agent는 독립적으로 global network를 업데이트시켰습니다. 이 때, 각 worker는 다른 worker의 학습중인 parameter가 반영된 global network의 parameter를 활용해 비동기적인 학습을 하였다고 했습니다. 다른 worker로부터 영향을 받은 parameter는 optimal이 아닐 수 있기 때문에, 이러한 parameter로 업데이트하는 방식인 A3C의 worker들은 학습의 *inconsistency*를 어쩔 수 없이 경험할 수 밖에 없었습니다. 

**이러한 개별 worker 학습의 inconsistency 문제를 해결하기 위해서, A2C의 coordinator는 global parameter를 업데이트 하기 전에 모든 worker의 actor들이 global 업데이트 주기 혹은 종료조건(done)에 도달할 때 까지 기다린 후, 다음 episode에서 동일 policy로 각 worker의 actor가 행동합니다. 이러한 동기화된 gradient update는 학습과정을 더욱 응집화하고 빠른 수렴에 도달할 수 있도록 합니다.** 또한, A2C는 A3C 대비 더욱 큰 규모의 배치 사이즈에 대해서도 학습을 더 잘다는 장점이 있습니다. 아래는 A3C와 A2C의 비교 그림입니다.

![Alt Text](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/Equation_img/3-4.A2C/a3c_a2c.png)

## 3-6. DPG (Deterministic Policy Gradient)([논문](https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/6.%20DPG/DPG.pdf)|[코드]())
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

*Todo (9/19)*
- [x] 0. Preliminary
- [x] 1. Value-based reinforcement learning
- [x] 2. Introduction and goal of *Policy gradient*
- [x] 3. Policy gradient algorithm
  - [x] 3-1. REINFORCE
  - [x] 3-2. Actor-critic
  - [x] 3-3. Off-policy policy gradient
  - [x] 3-4. A3C
  - [x] 3-5. A2C
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
