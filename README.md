# -RL-PolicyGradient_summarization
이 저장소는 **강화학습(RL, Reinforcement learning)** 의 학습 방법 중 하나인 **정책 경사(***policy gradient***)** 에 대한 방법론을 정리합니다.
원문은 다음 링크 >https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html 을 참고하였음을 밝힙니다.

또한 본 저장소는 **policy gradient**에 대한 전반적인 내용을 정리함과 더불어, 다양한 **policy gradient** 기반의 심층 강화학습 알고리즘들을 직접 구현하고, 해당 소스코드를 공유하는 것을 목적으로 만들어졌습니다.
저의 논문 작성과 연구를 위한 작은 저장소가 많은 사람들에게 도움이 되었으면 합니다 :)

# Value-based RL vs. Policy-based RL
고전적 강화학습 알고리즘들은 어떤 상태에서 취할 수 있는 행동별 가치를 나타내는 action-value function인 Q-function을 approximate/estimate하고, 해당 Q-function을 활용해 특정 상태, 행동 pair에 대한 값들을 계산/비교함으로써 \epsilon greedy 와 같은 방법으로 agent가 최적의 행동을 수행할 수 있도록 하였습니다.
즉, Q-function이라는 **어떤 상태(s)에서 취할 수 있는 행동(a)별 가치인 Q(s, a)** 를 계산함으로써, agent가 일련의 의사결정(sequential decision making)

강화학습에 대한 기본적인 사항이 숙지가 필요하다면 https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#key-concepts 링크를 먼저 참고하시면 좋습니다.

Richard S. Sutton 교수님의 policy gradient 논문 : http://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
