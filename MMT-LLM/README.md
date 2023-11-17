---
title: Multilingual Machine Translation with Large Language Models
author: Anusha Rao, Sanjana Moudgalya
date: 2023-11-16
category:
 - MT
tag:
 - Multilingual MT
 - Large Language Models
 - Evaluation with BLEU
star: true
---

# Multilingual Machine Translation with Large Language Models

To all those tech enthusiasts out there, here’s yet another advancement of Large Language Models (LLM) that’s worth a read. This article explores how powerful these models are in language translation tasks by presenting a summary of evaluations on 8 LLMs including but not limited to Chat-GPT, GPT-4, and Google Translate.

<!-- more -->

Paper: <https://arxiv.org/pdf/2304.04675.pdf>

Code: <https://github.com/NJUNLP/MMT-LLM>

## Introduction

![alt text](https://github.com/sanjana-moudgalya/mnlp_blog/blob/main/MMT-LLM/Introduction.png?raw=true)
Effective communication is universally recognized as a key to success. But what about the challenges posed by linguistic barriers? Imagine being unable to articulate your thoughts without understanding the language of the person you're talking to. Wouldn't it be incredible if language was no longer a barrier, enabling seamless communication with anyone across the globe? Sounds crazy, right? But might not be that far away from reality [Figure 1], and all credit to the advancement in Large Language Models(LLMs).

Before we delve deeper into translation tasks, let's first understand what LLMs are. Although you might not be familiar with their workings or design, chances are you've encountered them in various forms. For instance, Chat-GPT - does this sound familiar? They are smart systems powered by AI that have learned from heaps of text from all around the world, making them one of the most powerful models out there [Figure 2]. Their training corpus consists of multiple languages, allowing them to master not only their grammar but also the ability to understand the semantics of a language. Thus making them ideal for Multilingual Machine Translation (MMT) tasks.

![alt text](https://github.com/sanjana-moudgalya/mnlp_blog/blob/main/MMT-LLM/LLMs.png?raw=true)
Now that we know what these models are capable of, let's evaluate their true abilities in language translation. Based on the base paper, we attempt to answer two main questions. Firstly, how effectively are LLMs able to translate text across numerous languages? For languages like English, it is very easy to find training data since it is widely spoken. But what about languages like Quechua that neither have a large amount of literary works, nor a significant number of native speakers? Are the models still able to generate reasonable translations? Secondly, what factors significantly influence the performance of LLMs in translation tasks? Is it to do with the prompts that we feed in or is it to do with the training data? 

As per the results reported in the base paper, they perform a series of assessments on 8 LLMs that include English-centric ones like OPT, LLaMA2, LLaMA2-Chat, Falcon, and multilingual LLMs such as XGLM, BLOOMZ, ChatGPT, and GPT-4. It considers a broad spectrum of 102 languages across 606 translation directions. Their findings reveal improvements in the multilingual translation capabilities of LLMs, with GPT-4 achieving considerably high performance. Throughout this article, we focus on the working patterns of LLMs uncovered by the base paper and present our conclusions on the strengths and weaknesses of LLMs on MMT tasks. 

## Limitations with the Existing State of Machine Translation using LLMs

While the study highlights the impressive capabilities of LLMs, it's essential to acknowledge certain challenges and issues associated with machine translation systems, particularly the ones that use LLMs. One main challenge is that LLMs exhibit unbalanced translation capabilities across languages. They tend to perform better in translating into English than into non-English languages. Furthermore, GPT-4 faces greater challenges in French-centric and Chinese-centric translations, emphasizing the need for more balanced capabilities. 

Another main challenge is the challenges caused by the underlying representations of language models. They can unknowingly propagate biases present in their training data to downstream tasks. If the training data contains biases, the language model may exhibit biased behavior, potentially reinforcing stereotypes or prejudices. Despite their linguistic power, language models may lack true comprehension and understanding. They generate responses based on patterns learned during training but may not truly grasp the underlying meaning. This can lead to responses that sound plausible but are factually incorrect or nonsensical.

Lastly, language models heavily rely on the data they are trained on. If there are language subtleties or cultural contexts not well-represented in the training data, the model may struggle with accurate translations in those areas. In the context of handling data, these models may inadvertently manage sensitive information improperly. There's a risk that language models might generate responses that reveal private or sensitive information, raising concerns about privacy and data security.

Addressing these challenges is crucial for the responsible development and deployment of large language models, ensuring they positively contribute to diverse applications without unintended consequences or biases.

## Methodology

The primary focus of any MMT task is to gather data, with Flores-101 being a popular choice, as expressed by most researchers. No different from this is the base paper’s strategy to benchmark the LLMs on the Flores-101 dataset, assessing various model qualities across a wide array of languages. This dataset is specifically curated for language translation tasks and it consists of source-target pairs across 101 languages, including low-resource languages such as Swahili and Quechua. 

![alt text](https://github.com/sanjana-moudgalya/mnlp_blog/blob/main/MMT-LLM/ICL.png?raw=true)
To evaluate the model’s performance, the base paper employs the popular method of in-context learning (ICL). This is a technique used in machine learning where a system learns and improves by studying and understanding specific examples provided in the prompt[Figure 3]. This can be related to how humans learn in a real-world situation by observing examples of the task. For example, let’s consider one wants to learn how to ride a bicycle. Traditional learning might involve reading a manual or watching videos on how to ride a bicycle. But according to ICL, you learn by actually getting on a bicycle and practicing in a safe and open area. Initially, you might struggle with balance, but with practice, you gradually understand how to maintain stability, pedal, steer, and eventually ride confidently. Similarly, this study uses ICL and considers various factors, including in-context examples, templates, and the impact of different languages on translation.

Using this strategy, the authors compare the performance of eight different LLMs (OPT, LLaMA2, LLaMA2-Chat, Falcon, XGLM, BLOOMZ, ChatGPT, and GPT-4) with three supervised baselines (M2M-100, NLLB, Google Translator). While LLMs excel in machine translation due to their diverse training data and flexible architectures, enabling them to grasp complex linguistic features, supervised baselines rely on curated datasets or specific task-oriented training, making them more suitable for MMT. In this paper, the authors explore the relationship between translation performance and the pre-training corpus size of LLMs, uncovering some fascinating results.  

Now that we have an understanding of the language models and the translation task, we will look into methods that might be useful to evaluate the task.  While a commonly used score to evaluate MMT models is the BLEU (Bilingual Evaluation Understudy) score, an alternative to this with fairer evaluation for low-resource languages is the SpBLEU score. It counts matching words/phrases between the translated sentences and reference sentences. It then calculates precision based on these matches and penalizes the length to ensure a reasonable length of translations. In addition to these steps, SpBLEU uses text that is tokenized using the language-independent SentencePiece library. On the other hand, COMET and SEScore use a neural network to evaluate the translation to emphasize semantic similarity and sentence-level context. This article focuses on the three above-mentioned metrics, SpBLEU, COMET, and SEScore to provide a comprehensive comparison of LLMs with strong supervised baselines to reveal the gap between translation paradigms.


## Analysis of using LLMs for Machine Translation

An in-depth analysis of the performance of various LLMs reveals key factors influencing their translation performance. The study focuses on resource efficiency, prompt template design, and the importance of cross-lingual examples. Here are the main findings:

#### 1. Pre-training Corpus Size: 
LLMs can perform moderately well in a resource-efficient manner even on low-resource languages, like Catalan and Swahili. Even if the translation capabilities aren’t as great as high-resource languages, LLMs demonstrate a great learning potential through In-Context Learning (ICL), highlighting the model's adaptability.

#### 2. In-context Template:
The choice of prompt template significantly impacts translation performance, with variations of up to 16 BLEU points. Surprisingly, even seemingly unreasonable templates can effectively guide LLM in generating quality translations. "X = Y" results in the highest average BLEU score, whereas "[SRC]: X \n [TGT]: Y" achieves the lowest score, where X is the source sentence and Y is the target sentence. This shows us how dynamic seemingly similar templates can be with respect to performance. <br>
Example of the prompt template that gave us the best results during reproduction: <br>
"X = Y" format: „Wir haben jetzt 4 Monate alte Mäuse, die Diabetes hatten und jetzt keinen mehr haben“, fügte er hinzu. = "We now have 4-month-old mice that are non-diabetic that used to be diabetic," he added.

#### 3. Cross-lingual Exemplar:
![alt text](https://github.com/sanjana-moudgalya/mnlp_blog/blob/main/MMT-LLM/Cross-lingual%20Examplars.png?raw=true)
Cross-lingual examples in the prompts prove beneficial for certain translation directions, especially for those involving low-resource languages (such as Quechua-English), showcasing potential versatility. The selection of semantically related examples does not necessarily enhance performance compared to randomly picked examples. LLM learns core translation features through examples, emphasizing the importance of context and diversity. However, it is also important to notice the correlation of performance with the number of examples provided. BLEU score improves up to 8 examples, plateaus till 32, and then gradually declines [Figure 4].

#### 4. Translation Granularity: 
Word-level and document-level examples negatively affect LLMs’ performance, emphasizing the need for appropriate granularity in example selection. Diverse and contextually relevant examples contribute to better translation outcomes. For instance, consider a translation task where a document-level example is used to illustrate the context of a specific phrase. If this lengthy document contains various unrelated topics, the model might struggle to extract the relevant information for the translation, leading to inaccuracies or inconsistencies in the output. Conversely, a word-level example precisely targeting the phrase's context allows the model to focus on the specific linguistic nuances needed for accurate translation.

#### 5. Prompt Structure Impact:
The placement of examples within the prompt has a varying impact on LLM's behavior.
Reversing examples at the end of the prompt consistently leads to poorer results, highlighting the significance of prompt structure. For instance, compare these two prompts: "Explain the impact of prompt structure on LLM. Provide examples to illustrate your points." versus "Offer examples that demonstrate how LLM's behavior varies due to prompt structure. Explain the importance of prompt structure." In the first prompt, by asking for examples after the directive to explain, the model receives a clear context for the examples, enabling it to generate more coherent and relevant responses. Conversely, the second prompt, requesting examples before an explanation, poses a challenge for the model to understand the context for those examples, often resulting in less cohesive or relevant outputs.

## Summary

In the broader context of multilingual machine translation, this paper evaluates popular LLMs, such as ChatGPT and GPT-4, on 102 languages and 606 directions. While acknowledging continuous improvements, challenges remain for low-resource languages. LLMs have a lot of strengths, including the ability to ignore instruction semantics during in-context learning and the effectiveness of cross-lingual examples for low-resource translations. The analysis suggests a promising future for LLMs in resource-efficient multilingual machine translation.

The authors of this paper have done a good job at maintaining the codebase. Their GitHub repository is reproducible with a fairly small amount of changes. To provide evidence to the limitations and analysis section in this blog, we have a few BLEU scores after reproducing this paper using the NLLB (600M parameter) model -

German-English > 43.78426080060355 <br>
Assamese-English > 27.894121883466795 <br>
English-Assamese > 23.10996837152803 <br>
Swahili-English > 39.080203011245224 <br>

We can see that the BLEU score is lesser for Assamese and Swahili (low-resource languages when compared to German). Another important point to note is that generating Assamese text results in a lower score compared to generating English text, mainly because learning the vocabulary (to generate, instead of interpret) of a low-resource language might be harder.

## References
[1] Zhu, Wenhao, et al. "Multilingual machine translation with large language models: Empirical results and analysis." arXiv preprint arXiv:2304.04675 (2023) <br>
[2] https://thegradient.pub/in-context-learning-in-context/  <br>
[3] https://blog.ml6.eu/navigating-ethical-considerations-developing-and-deploying-large-language-models-llms-d44f3fcde626  <br>
[4] https://www.fiverr.com/ilovhus/translate-you-from-a-language-you-choose-to-any-other-language  <br>
