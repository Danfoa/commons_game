# The Commons Game

This repository presents a custom implementation of game in the paper 
[A multi-agent reinforcement learning model of common-pool resource appropriation](http://papers.nips.cc/paper/6955-a-multi-agent-reinforcement-learning-model-of-common-pool-resource-appropriation)
 by Deep Mind, presented at [Advances in Neural Information Processing Systems 30 (NIPS 2017)](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017).


The main goal of the paper is to introduce Reinforcement Learning based simulated environments as a way of addressing the 
modeling of common-pool resource dynamics. This, because abstract models based on non-cooperative game theory fail to predict 
deal world dynamics of these scenarios.

![Capture](https://user-images.githubusercontent.com/8356912/82810942-f2d86000-9e8f-11ea-9a3f-2f898842713b.PNG)

## Details

In this implementation we replaced the original DQN algorithm of the original paper with DDQN with 
replay buffer. Furthermore, we dont use a MLP architecture but rather a light weight CNN that could theoretically 
process the agents observations easily. More information on the implementation can be found in the following example Colab.

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/drive/1og_cjNZ9-mii5-ljlFpoAMSDa1iXehJU?usp=sharing">
    <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
    See details in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/Danfoa/commons_game">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub</a>
  </td>
</table>

## Requirements

The only requirements of the project are 
*    [Tensoflow2.x](https://www.tensorflow.org/install) 
*    [OpenGym python package](https://gym.openai.com/) 

Additionally, if you want to go ahead and see the code functionality first, you can check the example Colab notebook.

## TODOs

*    Use and compare novel RL value-based and policy-gradient based algorithms. 
*    


## Acknowledgements 
This repository is based on the code of: 

*    [sequential_social_dilemma_games](https://github.com/eugenevinitsky/sequential_social_dilemma_games)
*    [TensorFlow2.0-for-Deep-Reinforcement-Learning](https://github.com/Huixxi/TensorFlow2.0-for-Deep-Reinforcement-Learning)

