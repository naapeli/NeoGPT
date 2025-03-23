# NeoGPT

NeoGPT is a project designed to recreate OpenAI's GPT-2 from scratch, following its general architecture and design principles while incorporating custom modifications during the training phase. The primary goal is to explore the inner workings of transformer models, particularly how they scale with data and computational resources. At its core, NeoGPT utilizes the transformer architecture, which relies on self-attention mechanisms to process sequences of data efficiently. This architecture has become a cornerstone for many state-of-the-art models in natural language processing (NLP). By recreating GPT-2, this project not only serves as a technical exercise but also provides insights into the challenges and trade-offs involved in training large-scale models from scratch.

The model was trained using the Fineweb-edu dataset, which was chosen for its extensive coverage and suitability for language modeling tasks. The training process was conducted on Lambda Labs' GPU cluster, leveraging powerful NVIDIA GPUs that were crucial for handling the large-scale computations. Unfortunately, due to the high cost of extended GPU usage, training was paused after completing only one epoch. While this limited training session provided some insights, longer training could yield further improvements. The inspiration for NeoGPT comes from a tutorial by Andrej Karpathy, which outlines the process of building a GPT-like model from scratch.

The images below display the training metrics, offering a snapshot of the model's progress. The first image illustrates the loss function over time, a standard indicator of the model's learning efficiency. Notable spikes in the training loss at 5000 and 15000 steps are visible, resulting from starting the training process from model snapshots. These spikes could have been avoided had the optimizer and dataloader states been saved in addition to the model parameters. Furthermore, at 5000 steps, the implementation of the dataloader was modified to shuffle both the dataset shards and the tokens within each shard, causing a jump in the training loss. This shift was necessary due to the original dataset being insufficiently uniformly ordered. The second image tracks the model's performance on the Hellaswag metric, which evaluates its commonsense reasoning capabilities. These metrics offer an indication of the model's performance during the training.

![Image of the loss function as a function of training step](Images/Loss.png)  
*Figure 1: Loss function as a function of training steps.*

![Image of the hellaswag metric as a function of training step](Images/Hellaswag.png)  
*Figure 2: Hellaswag metric as a function of training steps.*

In Figure 1, we see that the model is able to surpass the performance of the original GPT2 when it comes to the loss. In Figure 2, we see that the model is not able to perform as well as GPT2 124M in a reasoning task. Hence futher improvements can be made by training the model longer.

Below are some completions of the sentence "Hello, I'm a language model," produced by the trained model. As observed, the model is generally able to generate grammatically correct sentences, but the coherence and relevance to the intended context varies a lot.

- *Hello, I'm a language model, and I'm sure it will get over the edge, depending on the context we're working on. If the*
- *Hello, I'm a language model, and I'm not an academic, I'm a programmer.’\nIn today's article, I'm*
- *Hello, I'm a language model, but I'm aware that when I say something I'm describing in a way that you know the way I say*
- *Hello, I'm a language model, because I'm learning Java.\nSo, what do we have in our code? It is an abstraction of*
- *Hello, I'm a language model, so you haven't heard of languages from the past century to the present. And not all languages were created equally*


Despite the early termination of the training due to computational constraints, this project has provided valuable learnings about the challenges and considerations of building and training large-scale language models. Moving forward, the plan is to refine and fine-tune the model by applying techniques such as Parameter-Efficient Fine-Tuning (PEFT) to enhance its conversational capabilities. With more affordable computational resources, the pretraining can be resumed and expanded, offering the potential for further improvements and more advanced applications.
