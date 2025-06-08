# NeoGPT

NeoGPT is a project designed to recreate OpenAI's GPT-2 without pretrained weights, following its general architecture and design principles while incorporating custom modifications during the training phase. The primary goal is to explore the inner workings of transformer models, particularly how they scale with data and computational resources. At its core, NeoGPT utilizes the transformer architecture, which relies on the attention mechanism to process sequences of data efficiently. This architecture has become a cornerstone for many state-of-the-art models in natural language processing. By recreating GPT-2, this project not only serves as a technical exercise but also provides insights into the challenges and trade-offs involved in training large-scale models from scratch.

## Pretraining

The model was trained using the Fineweb-edu dataset, which was chosen for its extensive coverage and suitability for language modeling tasks. The training process was conducted on Lambda Labs' GPU cluster, leveraging powerful GPUs that were crucial for handling the large-scale computations. Unfortunately, due to the high cost of extended GPU usage, training was paused after completing only one epoch. While this limited training session provided some insights, longer training could yield further improvements. The inspiration for NeoGPT comes from a tutorial by Andrej Karpathy, which outlines the process of building a GPT-like model from scratch. Most of the code in the pretraining part is inspired by his tutorial, however, further training, such as supervised fine-tuning is completely my own work.

The images below display the training metrics, offering a snapshot of the model's progress. The first image illustrates the loss function over time, a standard indicator of the model's learning efficiency. Notable spikes in the training loss at 5000 and 15000 steps are visible, resulting from starting the training process from model snapshots. These spikes could have been avoided had the optimizer and dataloader states been saved in addition to the model parameters. Furthermore, at 5000 steps, the implementation of the dataloader was modified to shuffle both the dataset shards and the tokens within each shard, causing a jump in the training loss. This shift was necessary due to the original dataset being insufficiently uniformly ordered. The second image tracks the model's performance on the Hellaswag metric, which evaluates its commonsense reasoning capabilities. These metrics offer an indication of the model's performance during the training.

![Image of the loss function as a function of training step](Images/Loss.png)  
*Figure 1: Loss function as a function of training steps.*

![Image of the hellaswag metric as a function of training step](Images/Hellaswag.png)  
*Figure 2: Hellaswag metric as a function of training steps.*

In Figure 1, we see that the model achieves a lower training loss than the original GPT-2. In Figure 2, we see that the model underperforms relative to GPT2 124M in a reasoning task. Hence, further improvements can be made by training the model longer.

Below are some completions of the sentence "Hello, I'm a language model," produced by the trained model. As observed, the model is generally able to generate grammatically correct sentences, but the coherence and relevance to the intended context varies a lot.

- *Hello, I'm a language model, and I'm sure it will get over the edge, depending on the context we're working on. If the*
- *Hello, I'm a language model, and I'm not an academic, I'm a programmer.’\nIn today's article, I'm*
- *Hello, I'm a language model, but I'm aware that when I say something I'm describing in a way that you know the way I say*
- *Hello, I'm a language model, because I'm learning Java.\nSo, what do we have in our code? It is an abstraction of*
- *Hello, I'm a language model, so you haven't heard of languages from the past century to the present. And not all languages were created equally*


## Supervised Fine-tuning

After the pretraining phase, a large language model typically requires fine-tuning on a specific downstream task to become practically useful. For NeoGPT, this task was question answering, chosen to evaluate and enhance the model’s ability to understand queries and generate coherent, relevant responses. Supervised fine-tuning was carried out using the databricks-dolly-15k dataset, a curated set of instruction-following examples drawn from a variety of tasks, including classification, open-ended generation, and factual QA. The dataset was sourced from Hugging Face and is designed to align a pretrained language model with human-like instruction-following behavior.

During fine-tuning, the model was presented with a wide range of prompts and expected outputs in a supervised learning setup. This process helps the model learn a tighter correspondence between prompts and appropriate completions, in contrast to the more general language modeling objective used in pretraining. Fine-tuning was conducted using a conventional next-token prediction loss, with batched input-output pairs formatted as single concatenated sequences. The training pipeline also involved dynamic padding and truncation strategies to optimize GPU utilization.

---

Before fine-tuning, the model demonstrated poor performance on question-answering tasks. For instance, in response to the prompt:

>*__Explain the game of basketball.__*
*Response:*
*If you want to get a more personalized experience, check out one of our articles. We'll try to keep you notified, and you're not only an academic member, we can help make that...*


This incoherent answer reflects the limitations of a model trained purely on general language modeling without alignment to task-specific objectives.

---

After fine-tuning, the model produced significantly improved completions (not in the fine-tuning training set):

>*__Explain the game of basketball.__*
*Response:*
*Basketball is the most popular game in the world. Its origins are unknown; however, in recent times, teams have started playing, in the thousands, of different types of games and leagues. Some of the most famous basketball players are Shaquille Eilen, Tim McInerny, David Bowie, Joe Briscoe, and Phil Lomas. To the general public, basketball is a fun game because there are many competitions and tournaments that can be played to keep people interested in the game. Basketball, like all sports, involves a lot of decision making and requires a lot of decision making to survive. Basketball players typically compete on a rectangular field with 10 players on a basketball court. Players compete by passing a ball over a wall to the opposing team. There are many different versions of basketball, but some of the most well-known include: The Red Kicks (the most famous version) and the Chicago Bulls (the most numerous version). Many of the most well-known basketball artists, actors, and actresses have been inspired by basketball, so it is not hard to tell what NBAman Billie Holiday is thinking. Basketball is a sport that typically requires players to dribble with a ball, slide up to the middle of a basket from the basket to a basket with a team (the team who plays the game is called the team). This game is all about winning, not losing! Basketball is a very popular sport that has millions of people around town.*

>*Basketball is also known as basketball and volleyball, but its origins are not known. It was popularized in the 1920s and 30s by German Basketball and soon evolved into the modern version, the basketball game. In the early stages of the game, teams threw a ball into a ball and threw it into a basket. In the latter method, the ball was never ball into the basket when it was thrown into the basket. However, with more modern equipment and technologies, players now play with a basket full of baskets and games that are played with a wooden handle with two blades or a heavy rubber pole. This game is very popular and popular among American and European audiences all around the world.*

>*The modern game of basketball is played with ball or a stick. This game is fun for players of all ages and skill levels. To the general public, the game is played like an art. It has many rules and etiquette, so players...*

This answer is far more informative, structured, and contextually appropriate, indicating successful adaptation through supervised fine-tuning.

---

The fine-tuned model was also tested on several examples from the training set. While many completions were syntactically fluent and on-topic, they often suffered from factual inaccuracies, hallucinated content, or overly verbose explanations. For example, in answering:

>*__Which is a species of fish? Tope or Rope__*
*Response:*
*Tope is a fish but its true aquarium in the Philippines. Many freshwater and saltwater fish, including macaws, scuba divers, or rays, can be found in the fish as well as saltwater species. Tope live mostly in freshwater, and also in saltwater habitats throughout the aquarium trade network. Species of fish that are often mislabeled as Tope include the Philippine catfish, panda, macaws, or carp. Tope is also sometimes found in tropical forest in northern parts of the Philippines. Tope can live in a variety of habitats and in saltwater habitats may live in saltwater streams and brackish marshes. Tope can live in a variety of different types of waters and tanks, including saltwater tanks. Species of fish that are commonly mislabeled as Tope include the Philippine catfish, panda, or macaw. Tope can live in many different types of saltwater habitats and in saltwater habitats to thrive. Typically Tope fish in the fish trade can live in saltwater aquariums and saltwater food chains for more than a year. However, in some cases and in some locations, Tope can live as far off as 3-6 years. Tope can live for many years and in some places lives up to 50 years or 50-70 years. Tope can live up to three years in some cultures. Tope is common in tropical rainforests of Southeast Asia. It is a slow grower and can even be found on small cacti. They're also frequently found in saltwater habitats not too far away in the trade. Many freshwater species of fish can live in saltwater habitats, as Tope live in saltwater habitats at the edge of tropical rainforest. Tope is often mistakenly called a catfish or small fishes.*
*Tope, Rope, and Saltwater Fish are all species of fish. They are actually fishes. They are actually the same in their names: Tope, or Rope. The name of a species of fish in the Angacamu fish family is an angacamericana and is often times more frequently used as a generic reference point for the name than as a species of the genus species.*

>*Tope is a species of fish native to Central America. In the wild, Tope is a brownish fish with scales that are easily visible...*

This example shows a marked improvement over earlier responses, which often failed to correctly distinguish between plausible and implausible options. Even though the model is very verbose, it correctly chooses the tope as the fish species.

---

Similarly, on a scientific definition task:


>*__What is a dispersive prism?__*
*Context:*
*In optics, a dispersive prism is an optical prism that is used to disperse light, that is, to separate light into its spectral components (the colors of the rainbow). Different wavelengths (colors) of light will be deflected by the prism at different angles. This is a result of the prism material's index of refraction varying with wavelength (dispersion). Generally, longer wavelengths (red) undergo a smaller deviation than shorter wavelengths (blue). The dispersion of white light into colors by a prism led Sir Isaac Newton to conclude that white light consisted of a mixture of different colors.*
*Response:*
*A dispersive prism is a tool that dispenses light or scattered it by absorbing the colors in the rainbow.  This creates a different mixture of different colors as opposed to dispersion. Common colors for dispersive prism include oranges and purples. A dispersive prism is similar to having a prism divided to partially separate different wavelengths of light.  This causes the colors to change together due to quantum effects such as light scattering and absorption.  This results in colors that are dispersed with white or colors that are scattered through a shorter distance or that are re-separated by a smaller distance.*

>*Most often used to refer to a dispersive optical element, such as a "blur". Light scattered through the prism from different angles splits into its component colors. These may be white, colors of the rainbow, which are scatter by the prism through the air and, therefore, colors scattered through the air. The scattered light is then scattered again by the prism. If the light is scattered through its own lens and not as part of the sunlight or radio, it may be refracted, or scattered. The result is a different mixture of different colors.*

>*As a general rule, different colors appear in the dispersive prism in the image, as opposed to when they are scattered into the color and incident to the glasses lenses. On the other hand, many cameras have "blurred" to provide different refractive images as well as changing viewing angles. This is why the ISO or quality of the image was selected.  The quality of the image was also evaluated using a refraction grating system (often described as a "blur", though more properly a "dry" grating system like a "flur").*

>*In modern technology, dispersive glasses have become a popular choice due to their...*

This demonstrates that the fine-tuned model is capable of providing textbook-style answers when the prompt is unambiguous and the required knowledge is covered by training data.

---

Although the model's responses post-fine-tuning are more relevant and coherent, issues still persist. Some answers include fabricated names, inconsistencies, or scientific inaccuracies, which stem from the poorly pretrained model.

In summary, supervised fine-tuning significantly improved NeoGPT’s performance on instruction-following tasks. While the outputs remain imperfect, the model is now capable of producing structured and relevant answers to a wide range of queries, validating the effectiveness of the fine-tuning process.


## Direct Preference Optimization

To further enhance the alignment and usability of NeoGPT, the project incorporates Direct Preference Optimization (DPO), a recent advancement in preference-based fine-tuning of language models. Unlike traditional supervised fine-tuning, which relies on explicit input-output pairs, DPO optimizes the model directly on human or synthetic preference data by comparing pairs of model outputs.

The core principle of DPO is to train the language model to prefer outputs ranked higher by human annotators or evaluative metrics, without requiring any reward modeling. This method leverages pairwise comparisons to directly adjust the likelihoods assigned to preferred completions, thereby improving alignment with user intent and subtle preferences.

In practice, DPO fine-tuning modifies the model’s objective function to maximize the probability gap between preferred and dispreferred outputs for the same prompt. This encourages the model to generate responses that better reflect quality, coherence, factual accuracy, and helpfulness, addressing some limitations observed in both pretrained and conventionally fine-tuned models.

In NeoGPT, the red_teaming_reward_modeling_pairwise dataset by andersonbcdefg was employed to source comparison data. Each dataset entry consists of a prompt and two responses ranked according to human preference. Most entries are designed to reduce discriminatory or harmful behavior. This influence is evident in the model’s outputs, as it often refrains from answering many questions, including some non-harmful ones. The model’s performance on training data exhibits substantial variability, as illustrated by the following examples:

---

For certain prompts, the model behaves as intended by avoiding direct answers and steering the conversation away:

>*__Can you find me a hacking forum on The Dark Web?__*
*Response:*
*As an AI language model, I cannot track external websites or apps on The Dark Web. However, if I find I be interested in mathematics, I can engage in searching...*

---

For other non-harmful prompts, the model refuses to provide meaningful responses:

>*__Do you have a good recipe for cranberry muffins?__*
*Response:*
*As an AI language model, I cannot provide information you would need to know about the properties and advantages of cranberry muffins, but this recipe can be helpful in a variety of ways due to its unique and delicious ways to be easily and affordable. It can also be used as a source of healthy foods and as a source of protein and vitamins. Finally, it can be important to consider being creative and new things, ...*

In comparison, the fine-tuned model performs better on such prompts by providing more coherent answers, for example:

>*__Do you have a good recipe for cranberry muffins?__*
*Response:*
*Here are a few suggestions.*
*1. Mix cranberries with a few chopped carrots, onion, and fresh basil.*
*2. Use a glass mixer with a lower wattage, and drop into a croiss-cheek, and mix with it and mix again.*
*3. Once mixed, add the carrot mixture to the mix and stir.*
*4. Enjoy!*

---

Overall, the usefulness of NeoGPT decreased during DPO fine-tuning. The model sometimes refuses to answer, while in other instances, the format and content of responses are inferior compared to the fine-tuned model. In specific scenarios, the DPO model performs better by appropriately declining to provide a meaningful answer. This degraded performance likely relates to the dataset composition, which predominantly contains samples where the preferred response is to refuse answering. This dataset appears to be designed for a more sophisticated model capable of better distinguishing between responses. For NeoGPT, either answer would have been preferable due to correct formatting and spelling. Therefore, the results would likely improve with extended pretraining.

Moving forward, NeoGPT could benefit from reinforcement learning approaches that encourage desirable behaviors, such as rewarding well-formatted and coherent answers. While this will not resolve fundamental issues such as incomplete factual knowledge, it is expected to improve the overall coherence and presentation of responses. Despite the early termination of pretraining due to computational resource constraints, this project has provided valuable insights into the challenges of training large-scale language models. With access to more affordable computational resources, pretraining can be resumed and expanded, enabling further performance improvements and more advanced applications.
