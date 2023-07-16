The main idea of this section is the introduction of a new method called Learning to Prompt for Continual Learning (L2P) that aims to train a more succinct memory system without accessing task identity at test time. The method involves using prompts, which are small learnable parameters maintained in a memory space, to instruct the model prediction and manage task-invariant and task-specific knowledge while maintaining model plasticity. L2P outperforms prior state-of-the-art methods in various challenging continual learning settings and can be applied without a rehearsal buffer, making it directly applicable to task-agnostic continual learning. The section also discusses the limitations of prior work in continual learning and introduces the use of prompts in prompt-based learning to address the challenges in retaining and utilizing learned knowledge without knowing task identity. L2P leverages pre-trained models and maintains a prompt pool, where prompts are dynamically looked up based on input features to guide the model's behavior for different tasks. This approach reduces catastrophic forgetting and allows for task-agnostic continual learning. The section concludes with the contributions of the proposed L2P method and the demonstrated effectiveness of L2P in various continual learning benchmarks.



In this section, the authors introduce Learning to Prompt for Continual Learning (L2P), a method that utilizes prompts to train a memory system in the field of continual learning. Unlike previous methods, L2P does not require a rehearsal buffer and achieves competitive results without it. The authors emphasize that their method provides a new perspective for addressing challenges in continual learning.

The section then discusses related work in continual learning, categorizing recent algorithms into regularization-based methods, rehearsal-based methods, and architecture-based methods. While rehearsal-based methods achieve state-of-the-art performance, their effectiveness deteriorates with smaller buffer sizes and they are not applicable in scenarios where data privacy is a concern. L2P, on the other hand, stores past knowledge in small learnable prompt parameters instead of using a rehearsal buffer, achieving performance similar to rehearsal-based methods without the need for task identity at test time.

The authors also mention recent work related to knowledge management and transfer learning, highlighting the use of prompting in these areas. They explain that prompting is designed for transfer learning, but it is non-trivial to apply it directly to continual learning. L2P's novel framework reveals the value of prompts in solving continual learning problems.

Continual learning protocols are discussed next, with a focus on different settings and challenges, such as task-incremental, class-incremental, domain-incremental, and task-agnostic settings. L2P specifically tackles the more challenging class-incremental and domain-incremental settings, as well as explores the task-agnostic setting.

Prompt-based learning and baselines are also mentioned, with a description of Prompt Tuning (PT) as an example technique in NLP. The authors explain that in PT, task-specific prompt functions are used to instruct pre-trained models for down-stream NLP tasks. However, prompting cannot be directly applied to continual learning scenarios where test-time task identity is unknown.

Overall, this section introduces the L2P method, discusses related work in continual learning, prompts in transfer learning, continual learning protocols, and the limitations of prompting in continual learning scenarios with unknown task identity.



In this section, the authors introduce the concept of a prompt pool and its motivations. They explain that the prompt pool is used to store encoded knowledge and is composed of individual prompts that can be flexibly grouped as an input to the model. The goal is to achieve fine-grained knowledge sharing through prompt combinations at the instance-wise level, where similar inputs tend to share more common prompts.

Next, the authors discuss the instance-wise prompt query mechanism, which dynamically selects suitable prompts for different inputs. They design a key-value pair based query strategy, where each prompt is associated with a learnable key. The input instance decides which prompts to choose through query-key matching. The purpose of this strategy is to decouple the query mechanism learning and prompt learning processes while making the framework task-agnostic.

The authors also mention an optional extension to the framework that adds task boundary prior information, which helps the model learn better task-specific prompts, especially when tasks have high diversity. This extension involves maintaining a prompt frequency table and penalizing frequently-used prompts to encourage diversified prompt selection.

In terms of the optimization objective for L2P, at each training step, N prompts are selected using the query strategy, and the adapted embedding feature is fed into the pre-trained model and a final classifier. The authors aim to minimize the end-to-end training loss, which includes a softmax cross-entropy loss and a surrogate loss to pull selected keys closer to corresponding query features.

Moving on to the experiments, the authors evaluate L2P in various settings, including class-incremental, domain-incremental, and task-agnostic settings. They compare L2P against several baseline methods and state-of-the-art rehearsal-based methods, conducting comprehensive experiments and providing ablation studies for a deeper understanding of their method.

In summary, this section focuses on the prompt pool, the instance-wise prompt query mechanism, the optimization objective for L2P, and the experimental evaluation of the method.



In this section, the authors discuss the datasets and experimental details used to evaluate their proposed method, L2P. They mention using Split CIFAR-100, 5 datasets, CORe50, and Gaussian scheduled CIFAR-100 for class-incremental, domain-incremental, and task-agnostic settings. They explain the evaluation metrics used and provide training details for L2P.

The authors then present the main results of their experiments. They compare L2P against other methods in class-incremental, domain-incremental, and task-agnostic learning settings. They show that L2P consistently outperforms other methods in terms of average accuracy and forgetting. They also compare L2P to architecture-based methods and demonstrate its superior performance.

Next, the authors discuss the effectiveness of the core designs in L2P. They analyze the impact of prompt-related components such as the prompt pool design, learnable keys, and diversified prompt selection. They show that these components contribute to mitigating catastrophic forgetting and improving performance.

Lastly, the authors discuss the effect of hyperparameters in L2P, such as the size of the prompt pool, the length of a single prompt, and the selection size. They highlight the importance of choosing appropriate hyperparameter values for optimal performance.

In conclusion, this section provides an overview of the datasets used, presents the main experimental results comparing L2P to other methods, analyzes the effectiveness of core designs in L2P, and discusses the impact of hyperparameters on performance.



In this section, the authors present the main idea and contributions of their paper. They introduce L2P, a novel method for addressing the challenges of continual learning. L2P utilizes prompt-based learning and a shared prompt pool to adapt a pre-trained model to sequential tasks, effectively mitigating catastrophic forgetting. The authors demonstrate that L2P outperforms state-of-the-art methods in various continual learning problems, including class-incremental and domain-incremental settings. They also show that L2P is capable of handling even more challenging task-agnostic settings.

Additionally, the authors acknowledge and thank several individuals for their valuable discussions and provide a list of references for further reading.

Overall, the main focus of this section is to introduce L2P, highlight its advantages over existing methods, and provide a context for the rest of the paper.



Section A of the content discusses the potential negative societal impact of L2P. It mentions that if the pretrained model used in L2P has bias and fairness issues, those issues may be carried over during the continual learning process. It also highlights the need for testing the robustness of L2P in safety-critical applications and designing defense techniques to address potential security concerns.

Section B discusses the limitations of L2P. It mentions that while L2P has been demonstrated on vision models, it does not make assumptions about modalities and can be explored for other modalities in future work. It also highlights the need for more complex benchmarks to evaluate the ability of task-agnostic continual learning methods in realistic scenarios.

Section C provides details about the dataset used in the experiments, which is a split CIFAR-100 dataset. It explains that the dataset is split into 10 tasks, with each task containing 10 disjoint classes. It also mentions that CIFAR-100 is a challenging dataset for continual learning studies, especially in the class-incremental setting.

Overall, this section discusses the potential negative societal impact of L2P, its limitations, and provides information about the dataset used in the experiments.