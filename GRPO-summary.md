🏃From RLHF, PPO to GRPO

OpenAI popularized the concept of [RLHF](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback) (Reinforcement Learning from Human Feedback), where we train an **"agent"** to produce outputs to a question (the **state**) that are rated more useful by human beings.

<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FU3NH5rSkI17fysvnMJHJ%252FRLHF.png%3Falt%3Dmedia%26token%3D53625e98-2949-45d1-b650-c5a7313b18a0&width=768&dpr=4&quality=100&sign=877df9f8&sv=2" alt="RLHF" style="width:500px;"/>
</div>
<div style="text-align:center">RLHF simplified schema</div>

<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252Fn5N2OBGIqk1oPbR9gRKn%252FPPO.png%3Falt%3Dmedia%26token%3De9706260-6bee-4ef0-a7dc-f5f6d80471d5&width=768&dpr=4&quality=100&sign=1a8386bf&sv=2" alt="PPO" style="width:500px;"/>
</div>
<div style="text-align:center">PPO simplified schema</div>

In order to do RLHF, [**PPO**](https://en.wikipedia.org/wiki/Proximal_policy_optimization) (Proximal policy optimization) was developed. The **agent** is the language model in this case. 

<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FplVZSTOwKSQv5zQYjkge%252FPPO%2520formula.png%3Falt%3Dmedia%26token%3D8b1359c8-11d1-4ea8-91c0-cf4afe120166&width=768&dpr=4&quality=100&sign=2b15f552&sv=2" alt="PPO" style="width:500px;"/>
</div>
<div style="text-align:center">PPO formula. The clip(..., 1-e, 1+e) term is used to force PPO not to take too large changes. There is also a KL term with beta set to > 0 to force the model not to deviate too much away.</div>

In fact PPO approach is composed of 3 systems:

1. The **Generating Policy (current trained model)**
2. The **Reference Policy (original model)**
3. The **Value Model (average reward estimator)**

We use the **Reward Model** to calculate the reward for the current environment, and our goal is to **maximize this**!

The formula for PPO looks quite complicated because it was designed to be stable. 


<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FiQI4Yvv1KcvkK7g5V8vm%252FGRPO%2520%252B%2520RLVR.png%3Falt%3Dmedia%26token%3D2155a920-b986-4a08-871a-32b5bbcfdbe3&width=768&dpr=4&quality=100&sign=57fbc22d&sv=2" alt="PPO" style="width:500px;"/>
</div>
<div style="text-align:center">GRPO simplified schema</div>

DeepSeek developed [**GRPO**](https://unsloth.ai/blog/grpo) (Group Relative Policy Optimization) to train their R1 reasoning models. The key differences to PPO are:

1. The **Value Model is removed,** replaced with statistics from calling the reward model multiple times.
2. The **Reward Model is removed** and replaced with just custom reward function which **RLVR** can be used.

This means GRPO is extremely efficient. Previously PPO needed to train multiple models - now with the reward model and value model removed, we can save memory and speed up everything.

**RLVR (Reinforcement Learning with Verifiable Rewards)** allows us to reward the model based on tasks with easy to verify solutions. For example:

1. Maths equations can be easily verified. Eg 2+2 = 4.
2. Code output can be verified as having executed correctly or not.
3. Designing verifiable reward functions can be tough, and so most examples are math or code.
4. Use-cases for GRPO isn’t just for code or math—its reasoning process can enhance tasks like email automation, database retrieval, law, and medicine, greatly improving accuracy based on your dataset and reward function - the trick is to define a **rubric - ie a list of smaller verifiable rewards, and not a final all consuming singular reward.** OpenAI popularized this in their [reinforcement learning finetuning (RFT)](https://platform.openai.com/docs/guides/reinforcement-fine-tuning) offering for example.

**Why "Group Relative"?**

GRPO removes the value model entirely, but we still need to estimate the **"average reward"** given the current state.

The **trick is to sample the LLM**! We then calculate the average reward through statistics of the sampling process across multiple different questions.

<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FdXw9vYkjJaKFLTMx0Py6%252FGroup%2520Relative.png%3Falt%3Dmedia%26token%3D9153caf5-402e-414b-b5b4-79fef1a2c2fa&width=768&dpr=4&quality=100&sign=12ec7641&sv=2" alt="PPO" style="width:500px;"/>
</div>
<div style="text-align:center">"Group relative" logic</div>

For example for "What is 2+2?" we sample 4 times. We might get 4, 3, D, C. We then calculate the reward for each of these answers, then calculate the **average reward** and **standard deviation**, then **Z-score standardize** this!


<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FVDdKLOBcLyLC3dwF1Idd%252FStatistics.png%3Falt%3Dmedia%26token%3D6c8eae5b-b063-4f49-b896-7f8de516a379&width=768&dpr=4&quality=100&sign=6c940048&sv=2" alt="PPO" style="width:500px;"/>
</div>
<div style="text-align:center">GRPO advantage calculation</div>

This creates the **advantages A**, which we will use in replacement of the value model. This saves a lot of memory!

## 🤞Luck (well Patience) Is All You Need

The trick of RL is you need 2 things only:

1. A question or instruction eg "What is 2+2?" "Create a Flappy Bird game in Python"
2. A reward function and verifier to verify if the output is good or bad.

With only these 2, we can essentially **call a language model an infinite times** until we get a good answer. For example for "What is 2+2?", an untrained bad language model will output:

***0, cat, -10, 128, 3, A, B, 122, 17, 182, 172, A, C, BAHS, %$, #, 9, -192, 12.31*** ***then suddenly 4******.***

***The reward signal was 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0*** ***then suddenly 1.***

So by luck and by chance, RL managed to find the correct answer across multiple **rollouts**. Our goal is we want to see the good answer 4 more, and the rest (the bad answers) much less.

**So the goal of RL is to be patient - in the limit, if the probability of the correct answer is at least a small number (not zero), it's just a waiting game - you will 100% for sure encounter the correct answer in the limit.**

**So I like to call it as "Luck Is All You Need" for RL.**

**Well a better phrase is "Patience is All You Need" for RL.**

<div style="text-align:center">
<img src="https://docs.unsloth.ai/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FryuL3pCuF8pPIjPEASbx%252FLuck%2520is%2520all%2520you%2520need.png%3Falt%3Dmedia%26token%3D64d1a03a-6afc-49a9-b734-8ce8bc2b5ec1&width=768&dpr=4&quality=100&sign=13346422&sv=2" alt="PPO" style="width:400px;"/>
</div>

RL essentially provides us a trick - instead of simply waiting for infinity, we do get "bad signals" ie bad answers, and we can essentially "guide" the model to already try not generating bad solutions. This means although you waited very long for a "good" answer to pop up, the model already has been changed to try its best not to output bad answers.

In the "What is 2+2?" example - ***0, cat, -10, 1928, 3, A, B, 122, 17, 182, 172, A, C, BAHS, %$, #, 9, -192, 12.31*** ***then suddenly 4******.***

Since we got bad answers, RL will influence the model to try NOT to output bad answers. This means over time, we are carefully "pruning" or moving the model's output distribution away from bad answers. This means RL is not inefficient, since we are NOT just waiting for infinity, but we are actively trying to "push" the model to go as much as possible to the "correct answer space".

**If the probability is always 0, then RL will never work**. This is also why people like to do RL from an already instruction finetuned model, which can partially follow instructions reasonably well - this boosts the probability most likely above 0.

### Resources

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [Unsloth Reinforcement Learning Guide](https://docs.unsloth.ai/basics/reinforcement-learning-guide#grpo-notebooks)
