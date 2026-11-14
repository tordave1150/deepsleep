# แผนที่แนวคิด Deep Learning

[Index](../INDEX.md) · [Model Comparison](model-comparison.md)

```mermaid
flowchart TD
    T["Tensor และ PyTorch · 01"] --> M["MLP, Loss, Backpropagation · 02"]
    M --> E["Regularization และ Evaluation · 03"]
    M --> C["Convolution และ Feature Maps · 04"]
    E --> I["Transfer Learning และ Grad-CAM · 05"]
    C --> I
    M --> R["Hidden State และ BPTT · 06"]
    R --> G["LSTM และ GRU · 07"]
    R --> A["Attention และ Transformer · 08"]
    G --> A
    I --> S["Distillation, SSL และ Robustness · 09"]
    A --> S
    S --> Z["Latent Space และ Generative AI · 09"]
    C --> AE["Reconstruction และ VAE · 10"]
    Z --> AE
    AE --> GAN["Generator, Discriminator และ Metrics · 11"]
```

แผนภาพจัดทำจากลำดับการสอนและความรู้ที่บทหลังนำกลับมาใช้ ลูกศรแสดงความเชื่อมโยงเพื่อทบทวน เช่น Gradient จาก MLP ถูกใช้กับทั้ง BPTT และ Grad-CAM ส่วน CNN ให้ตัวอย่าง Feature Extraction ที่กลับมาใช้ใน Convolutional AE

- [พื้นฐาน MLP และ Evaluation](../lectures/02-mlp-basics.md)
- [ภาพและ Transfer Learning](../lectures/05-cnn-classifier.md)
- [ลำดับและ Attention](../lectures/08-attention-transformer.md)
- [Generative Models](../lectures/10-autoencoder.md)
