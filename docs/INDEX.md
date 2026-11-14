# สารบัญ Deep Learning

[README](../README.md) · [Lab Index](labs/README.md) · [สถานะ](STATUS.md)

## Lecture และ Lab ตาม Class

| Class | Lecture | หัวข้อหลัก | Lab |
| --- | --- | --- | --- |
| class01 | [Introduction และ PyTorch](lectures/01-introduction-pytorch.md) | Tensor, Hardware Acceleration, Computation Graph | [setup_pytorch](labs/setup_pytorch.md) |
| class02 | [MLP และวงจรการเรียนรู้](lectures/02-mlp-basics.md) | Forward Pass, CrossEntropy, Backpropagation, Optimizer | [MLP1](labs/MLP1.md) |
| class03 | [ปัญหาของ MLP และการติดตามการทดลอง](lectures/03-mlp-problems-visualization.md) | Batch Normalization, Regularization, Calibration, Experiment Tracking | [MLP2_ex1-1_StaticViz](labs/MLP2_ex1-1_StaticViz.md), [MLP2_ex1-2_TensorBoard](labs/MLP2_ex1-2_TensorBoard.md), [MLP2_ex1-3_WandB](labs/MLP2_ex1-3_WandB.md) |
| class04 | [Convolution และ CNN](lectures/04-cnn-basics.md) | Conv2d, Padding, Stride, Pooling, Spatial BatchNorm | [CNN1_ex2](labs/CNN1_ex2.md), [CNN1_ex3](labs/CNN1_ex3.md), [CNN1_ex4](labs/CNN1_ex4.md) |
| class05 | [CNN สำหรับ Image Classification](lectures/05-cnn-classifier.md) | Augmentation, Backbone, Fine-tuning, Grad-CAM | [CNN2_ex1](labs/CNN2_ex1.md), [CNN2_ex2](labs/CNN2_ex2.md), [CNN2_ex3](labs/CNN2_ex3.md), [CNN2_ex4](labs/CNN2_ex4.md) |
| class06 | [Recurrent Neural Network](lectures/06-rnn.md) | Hidden State, BPTT, Time Series, Teacher Forcing | [RNN_ex1_ex2_ex3](labs/RNN_ex1_ex2_ex3.md), [RNN_ex4_ex5](labs/RNN_ex4_ex5.md), [RNN_ex6](labs/RNN_ex6.md) |
| class07 | [LSTM, GRU และทางเลือกสำหรับข้อมูลลำดับ](lectures/07-lstm-gru.md) | Gates, Gradient Clipping, Padding, WaveNet | [LSTM_ex1](labs/LSTM_ex1.md), [LSTM_ex3](labs/LSTM_ex3.md) |
| class08 | [Attention และ Transformer](lectures/08-attention-transformer.md) | Q/K/V, Mask, Transformer, BERT, Mamba | [Attention_ex1](labs/Attention_ex1.md), [Attention_ex2](labs/Attention_ex2.md) |
| class09 | [Adversarial Attack และ Generative AI](lectures/09-adversarial-attack-generative-ai.md) | Distillation, Self-Supervision, Robustness, Latent Space | ไม่มี Notebook ในโฟลเดอร์ |
| class10 | [Autoencoder](lectures/10-autoencoder.md) | Reconstruction, KL, Reparameterization | [AE_ex1](labs/AE_ex1.md) |
| class11 | [Generative Adversarial Network](lectures/11-gan.md) | Minimax, Mode Collapse, FID, Conditional GAN | ไม่มี Notebook ในโฟลเดอร์ |

## ข้ามไปหัวข้อสำคัญ

- [Tensor](lectures/01-introduction-pytorch.md#tensor)
- [Training Loop](lectures/02-mlp-basics.md#training-loop)
- [Batch Normalization](lectures/03-mlp-problems-visualization.md#batch-normalization)
- [Spatial Convolution](lectures/04-cnn-basics.md#spatial-convolution)
- [Grad-CAM](lectures/05-cnn-classifier.md#grad-cam)
- [Vanilla RNN](lectures/06-rnn.md#recurrent-state)
- [LSTM Gates](lectures/07-lstm-gru.md#lstm-gates)
- [Scaled Dot-Product Attention](lectures/08-attention-transformer.md#attention)
- [Robustness](lectures/09-adversarial-attack-generative-ai.md#robustness)
- [KL และ Reparameterization](lectures/10-autoencoder.md#vae-loss)
- [Minimax Loss](lectures/11-gan.md#gan-objective)

## Review

- [Learning Roadmap](review/learning-roadmap.md)
- [Concept Map](review/concept-map.md)
- [Model Comparison](review/model-comparison.md)
- [Formula Reference](review/formula-reference.md)
- [Glossary](review/glossary.md)
- [Practice & Solutions](review/practice-and-solutions.md)

[Coverage และหลักฐานรายไฟล์](../sources/coverage.md)
