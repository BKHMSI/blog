---
layout: post
title: "Part 1: Episodic Meta-RL and the Brain"
date: 2020-01-21 00:00:00 +0300
description: This is the first part of a series of blog posts about my master’s dissertation.
img: neurons.jpg
tags: [Meta Learning, Reinforcement Learning, Meta-RL, Neuroscience]
---
 
This is the first part of a series of blog posts about my master’s dissertation. It assumes some preliminary knowledge in neuroscience and reinforcement learning, but fear not, I will be in the comments section if you have any questions. A bit about me first: I completed my MSc in Computational Cognitive Neuroscience (CCN) from Goldsmiths, University of London. Since I come from a Computer Science / Machine Learning background I decided to do my research project on work that lies in the intersection of both fields. I was largely inspired by the work done by the DeepMind neuroscience team on [meta-RL](https://deepmind.com/blog/article/prefrontal-cortex-meta-reinforcement-learning-system)<sup>20,21</sup> and Sam Ritter's PhD thesis<sup>22</sup> which this post is heavily based on. It was actually one of the main reasons I decided to join a master’s in CCN. It made me feel that it was necessary to understand the brain if one wishes to become a leading researcher in artificial intelligence. This post address the relation of episodic meta-RL to the brain, in an attempt of bridging the gap between machine learning and neuroscience. 

# Motivation 

In recent years, deep reinforcement learning (RL) has been at the forefront of artificial intelligence (AI) research, achieving breakthroughs in an array of different domains<sup>1,2,3</sup>. This led cognitive scientists to turn to those developments in AI and ask whether it can give us any insight concerning the mechanisms underlying human cognition<sup>21</sup>; considering that RL was originally inspired by psychological research in animal conditioning<sup>4,5</sup>. Take Ivan Pavlov’s seminal work as an example. In his experiments, dogs began salivating on the sound of a buzzer by associating it with food. Therefore they were able to predict the reward before actually observing it. Using this prediction signal one can train a dog or any animal in fact to perform a specific task. Similarly, RL algorithms can guide artificial agents through positive and negative rewards to perform a set of actions that maximize cumulative long-term reward.

However, one major shortcoming of deep-RL that automatically disqualifies it as a plausible model for biological learning is its sample-inefficiency<sup>21</sup>. In other words, the amount of experience an artificial agent requires in order to attain a suitable level of expertise is orders of magnitude more than what a human would need. To address this issue, Botvinick et al. (2019) identify two main sources that cause deep-RL to be slow; those are incremental parameter adjustment and starting with a weak inductive bias. The former is needed to avoid catastrophic inference while the latter allows the neural network to adapt to a wide range of problems. Taking that into consideration, they show that both factors can be mitigated by using an episodic memory and meta-learning on a distribution of learning tasks. They also note that both solutions had been shown to have further implications in neuroscience and psychology.

# Relation to Biology

## RL and the Striatal Dopamine System

First, let’s backtrack and see what traditional RL algorithms had to offer for neuroscience. It has been observed that the firing properties of [midbrain dopamine neurons](https://en.wikipedia.org/wiki/Dopaminergic_pathways) drive synaptic plasticity in the striatum to solidify the associations between experienced action and rewards<sup>6</sup>. Research in RL later led neuroscientists to develop a reward-based learning theory of dopaminergic function. Specifically, dopamine release encodes an index of ‘surprise’ that reflects reward prediction error (RPE) signals predicted by [temporal difference (TD)](https://en.wikipedia.org/wiki/Temporal_difference_learning) learning algorithms, or in other words the difference between the expected and actual reward<sup>7</sup>. Further research on non-human primates, humans, and rodents have converged on a canonical model in which the synchronous firing of pre and post-synaptic neurons leads to stronger synaptic connections in the presence of dopamine and weaker connections in its absence<sup>8</sup>. In other words, neuron firing that leads to an increase in RPE signal becomes reinforced by what is known as the “three-factor rule”<sup>9</sup>.

## Meta-RL and the Prefrontal Cortex

However, this canonical model has been put under strain by a number of findings in the [prefrontal cortex (PFC)](https://en.wikipedia.org/wiki/Prefrontal_cortex). It has been shown that sectors of the PFC encode quantities essential for RL such as expected values of actions and states<sup>10,11</sup>, as well as the recent history of rewards and actions<sup>12,13</sup>. This suggests that the PFC implements mechanisms for reward-based learning similar to those previously attributed to dopamine-based RL. Therefore, one question naturally arises: what’s the relationship between both systems? Wang et al. (2018) address this quandary by proposing a new theory in which the PFC and dopamine-based RL are two self-contained RL systems that implement different forms of learning. Specifically, they show that the PFC performs _model-based_ RL implemented in its activity dynamics by making use of representations of task structure as opposed to _model-free_ RL that is based on direct stimulus-response associations driven by dopamine synaptic learning. In this theory, the synaptic weights in the prefrontal network are shaped by dopamine-based RL over a series of interrelated tasks. Thus the model-free RL gives rise to a second independent model-based one that is capable of rapidly adapting to new environments, and is referred to as “meta-RL”. Wang et al. (2018) demonstrate through a number of simulations that meta-RL can explain a large range of behavioral and neurophysiological findings that presented difficulties for the standard dopamine-based model.

## Episodic Learning and the Hippocampus 
Ritter et al. (2018) point out that although the meta-RL framework proposed by Wang et al. (2018) provides a full account of incremental learning, it does not take into account episodic learning processes. They note that leveraging relevant past experience in order to inform the making of new decisions is an essential feature for any intelligent organism. Thus agents must also be able to distill the outcome of past decisions into memories and store them for long periods of time, with the ability to later retrieve the relevant ones when encountering similar contexts to appraise the value of future actions. Recent research had demonstrated that accounting for episodic learning, of the kind associated with the [hippocampus](https://en.wikipedia.org/wiki/Hippocampus), when analyzing human behavioral and fMRI data had led to better model-fits than only considering the incremental learning algorithms discussed before<sup>14</sup>. In addition to this, meta-RL agents had been shown to suffer from catastrophic forgetting<sup>15</sup>; which means that it has to re-learn strategies it had previously mastered instead of utilizing the knowledge it had gained from past experience. Those observations led to an increased interest in computational models that make use of episodic memory and demonstrated the importance of episodic learning processes in decision making<sup>16</sup>. Inspired by functional properties associated with the hippocampus, Ritter et al. (2018) design the extended architecture such that it includes features as pattern completion, pattern separation, and reinstatement of cortical activity patterns. This entails that memories should be retrieved and reinstated when the agent encounters a context similar to one it had seen before without fully interfering with the current state of its working memory.

# Episodic Meta-RL as a Computational Model
Following the work of Wang et al. (2018), the meta-RL model conceptualizes the PFC along with the subcortical structures to which it connects (i.e. [basal ganglia](https://en.wikipedia.org/wiki/Basal_ganglia) and [thalamic nuclei](https://en.wikipedia.org/wiki/List_of_thalamic_nuclei)) as forming a homogeneous recurrent neural network (RNN) which follows a nascent tradition of modeling this set of regions, abstracting many of the neurophysiological and anatomical details<sup>17</sup>. It is trained using an [actor-critic algorithm](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html), which reflects the dopamine-based learning system. The network receives as input at each time-step an encoded version of the perceptual data alongside a signal denoting the reward and the action it had taken at the previous time-step. The RNN learns to distill the history of observations, rewards, and actions into its hidden state, which can be considered as a form of working memory while training on a series of structurally interrelated tasks. Ritter et al. (2018) extends this architecture to include a non-parametric episodic memory implemented as a differentiable neural dictionary<sup>23</sup>. It stores the working memory states (i.e. the RNN hidden state) paired with a perceptual context embedding as key. This can later be used to retrieve and reinstate memory activations into the current working memory by using some learned gating mechanism when encountering a similar context. This model is in accord with what has been observed in neuroscientific studies; retrieving episodic memories triggers a pattern of activity similar to what has been originally experienced in circuits supporting working memory<sup>18</sup>. The reinstatement method employed in this model had also been shown to functionally resemble gating mechanisms operating in the PFC<sup>19</sup>. To sum up, the episodic meta-RL model is able to provide a unified account of model-free and model-based strategies when trained over a distribution of tasks that contain both episodic and incremental structure with the objective of maximizing a particular reward function.

# What’s Next
In the following post, I will use this model to reproduce the behavioral results of the episodic (or contextual) Two-Step task that is used to disassociate between a model-free and a model-based system. In the meantime you can find my implementation <a href="https://github.com/BKHMSI/Meta-RL-TwoStep-Task">here</a>.

# References

[1] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540):529–533.

[2] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbrenner, N., Sutskever, I., Lillicrap, T., Leach, M., Kavukcuoglu, K., Graepel, T., and Hassabis, D. (2016). Mastering the game of go with deep neural networks and tree search. Nature, 529:484–503.

[3] Vinyals, O., Babuschkin, I., Czarnecki, W., Mathieu, M., Dudzik, A., Chung, J., Choi, D., Powell, R., Ewalds, T., Georgiev, P., Oh, J., Horgan, D., Kroiss, M., Danihelka, I., Huang, A., Sifre, L., Cai, T., Agapiou, J., Jaderberg, M., and Silver, D. (2019). Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575.

[4] Pavlov, I. P. (1927). Conditioned reflexes: an investigation of the physiological activity of the cerebral cortex. Oxford University Press.

[5] Rescorla, R. and Wagner, A. (1972). A theory of Pavlovian conditioning: Variations in the effectiveness of reinforcement and nonreinforcement, volume Vol. 2.

[6] Schultz, W., Dayan, P., and Montague, P. R. (1997). A neural substrate of prediction and reward. Science, 275(5306):1593–1599.

[7] Sutton, R. S. and Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, USA.

[8] Daw, N. and Tobler, P. (2014). Value learning through reinforcement: The basics of dopamine and reinforcement learning. Neuroeconomics: Decision Making and The Brain, pages 283–298.

[9] Glimcher, P. W. (2011). Understanding dopamine and reinforcement learning: The dopamine reward prediction error hypothesis. Proceedings of the National Academy of Sciences, 108(Supplement 3):15647–15654.

[10] Plassmann, H., O’Doherty, J., and Rangel, A. (2007). Orbitofrontal cortex encodes willingness to pay in everyday economic transactions. Journal of Neuroscience, 27(37):9984–9988.

[11] Padoa Schioppa, C. and Assad, J. (2006). Padoa-schioppa, c. assad, j.a. neurons in orbitofrontal cortex encode economic value. nature 441, 223-226. Nature, 441:223–6.

[12] Seo, M., Lee, E., and Averbeck, B. (2012). Action selection and action value in frontal- striatal circuits. Neuron, 74:947–60.

[13] Tsutsui, K.-I., Grabenhorst, F., Kobayashi, S., and Schultz, W. (2016). A dynamic code for economic object valuation in prefrontal cortex neurons. Nature Communi- cations, 7:12554.

[14] Bornstein, A. M., Khaw, M. W., Shohamy, D., and Daw, N. D. (2017). Reminders of past choices bias decisions for reward in humans. bioRxiv.

[15] Ritter, S., Wang, J. X., Kurth-Nelson, Z., Jayakumar, S. M., Blundell, C., Pascanu, R., and Botvinick, M. M. (2018). Been there, done that: Meta-learning with episodic recall. In ICML.

[16] Gershman, S. J. and Daw, N. D. (2017). Reinforcement learning and episodic memory in humans and animals: An integrative framework. Annual Review of Psychology, 68(1):101–128. PMID: 27618944.

[17] Song, H. F., Yang, G. R., and Wang, X.-J. (2016). Reward-based training of recurrent neural networks for cognitive and value-based tasks. bioRxiv.

[18] Xiao, X., Dong, Q., Gao, J., Men, W., Poldrack, R. A., and Xue, G. (2017). Trans- formed neural pattern reinstatement during episodic memory retrieval. Journal of Neuroscience, 37(11):2986–2998.

[19] Chatham, C. and Badre, D. (2015). Multiple gates on working memory. Current Opin- ion in Behavioral Sciences, 1:23–31.

[20] Wang, J. X., Kurth-Nelson, Z., Kumaran, D., Tirumala, D., Soyer, H., Leibo, J. Z., Hassabis, D., and Botvinick, M. (2018). Prefrontal cortex as a meta-reinforcement learning system. bioRxiv.

[21] Botvinick, M., Ritter, S., Wang, J., Kurth-Nelson, Z., Blundell, C., and Hassabis, D. (2019). Reinforcement learning, fast and slow. Trends in Cognitive Sciences, 23.

[22] Ritter, S. (2019). Meta-reinforcement Learning with Episodic Recall: An Integrative Theory of Reward-Driven Learning. PhD thesis.

[23] Pritzel, A., Uria, B., Srinivasan, S., Badia, A. P., Vinyals, O., Hassabis, D., Wierstra, D., and Blundell, C. (2017). Neural episodic control. CoRR, abs/1703.01988.