# Deep Learning Knowledge Base

ตำราเรียนด้วยตนเองภาษาไทยสำหรับผู้เริ่มต้นและผู้ที่ต้องการทบทวนวิชา Deep Learning ใช้ศัพท์เทคนิคภาษาอังกฤษตามเอกสารเรียน และไล่เหตุผลจากภาพและสมการไปยัง PyTorch ทีละขั้น

## เริ่มอ่าน

เรียนตามลำดับด้านล่าง แต่ละบทมีลิงก์ไปยัง Lab ที่เกี่ยวข้อง แผนที่หัวข้อ และคำถามทบทวนพร้อมเฉลย ใน Lab ให้อ่าน data flow และลองทำนาย tensor shape ก่อนดู output ที่บันทึกไว้ หากอยากย้อนตรวจหลักฐาน ให้ใช้เลขหน้า PDF และเลข cell ที่ท้ายหัวข้อ

1. [Introduction และ PyTorch](docs/lectures/01-introduction-pytorch.md) · Tensor, Hardware Acceleration, Computation Graph
2. [MLP และวงจรการเรียนรู้](docs/lectures/02-mlp-basics.md) · Forward Pass, CrossEntropy, Backpropagation, Optimizer
3. [ปัญหาของ MLP และการติดตามการทดลอง](docs/lectures/03-mlp-problems-visualization.md) · Batch Normalization, Regularization, Calibration, Experiment Tracking
4. [Convolution และ CNN](docs/lectures/04-cnn-basics.md) · Conv2d, Padding, Stride, Pooling, Spatial BatchNorm
5. [CNN สำหรับ Image Classification](docs/lectures/05-cnn-classifier.md) · Augmentation, Backbone, Fine-tuning, Grad-CAM
6. [Recurrent Neural Network](docs/lectures/06-rnn.md) · Hidden State, BPTT, Time Series, Teacher Forcing
7. [LSTM, GRU และทางเลือกสำหรับข้อมูลลำดับ](docs/lectures/07-lstm-gru.md) · Gates, Gradient Clipping, Padding, WaveNet
8. [Attention และ Transformer](docs/lectures/08-attention-transformer.md) · Q/K/V, Mask, Transformer, BERT, Mamba
9. [Adversarial Attack และ Generative AI](docs/lectures/09-adversarial-attack-generative-ai.md) · Distillation, Self-Supervision, Robustness, Latent Space
10. [Autoencoder](docs/lectures/10-autoencoder.md) · Reconstruction, KL, Reparameterization
11. [Generative Adversarial Network](docs/lectures/11-gan.md) · Minimax, Mode Collapse, FID, Conditional GAN

## สารบัญและแบบทบทวน

- [สารบัญรายหัวข้อและ Lab](docs/INDEX.md)
- [Learning Roadmap](docs/review/learning-roadmap.md)
- [Concept Map](docs/review/concept-map.md)
- [Model Comparison](docs/review/model-comparison.md)
- [Formula Reference](docs/review/formula-reference.md)
- [Glossary](docs/review/glossary.md)
- [Practice & Solutions](docs/review/practice-and-solutions.md)

## ต้นฉบับและสิทธิ์การใช้งาน

ใช้เฉพาะ PDF และ Notebook ในโฟลเดอร์ Google Drive ของวิชา คัดลอกมาเฉพาะคำสั่งสั้น ๆ เพื่ออธิบายการทำงาน ภาพประกอบของ repository สร้างใหม่ ผู้เปิดต้นฉบับต้องมีสิทธิ์ใน Drive ตามที่เจ้าของกำหนด [โฟลเดอร์ต้นทาง](https://drive.google.com/drive/folders/1SQNa3jVmmPyw1Qvaz0fT26JxgkxrBmzK)

- [วิธีอ่านแหล่งที่มา](sources/README.md)
- [Coverage](sources/coverage.md) และ [Manifest](sources/manifest.json)
- [สถานะและผลตรวจ](docs/STATUS.md)
