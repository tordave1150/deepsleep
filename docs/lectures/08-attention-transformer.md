# 8. Attention และ Transformer

[← ก่อนหน้า](07-lstm-gru.md) | [Index](../INDEX.md) | [ถัดไป →](09-adversarial-attack-generative-ai.md)

## แหล่งเรียน

class08 · ตรวจต้นฉบับ 2026-11-14

- [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view)
- Lab [Attention_ex1](../labs/Attention_ex1.md)
- Lab [Attention_ex2](../labs/Attention_ex2.md)

## เป้าหมายและพื้นฐาน

คำนวณ Attention จาก Query, Key และ Value ตาม shape และอ่านบทบาทของ mask, positional information และชนิด Transformer ทวน Encoder–Decoder จาก [Class 06](06-rnn.md) กับการจัดลำดับและ Padding จาก [Class 07](07-lstm-gru.md)

## จาก Thought Vector ไปสู่ Attention

ใน Seq2Seq ที่ใช้ Thought Vector เดียว Decoder ต้องพึ่งสรุปความยาวคงที่เพื่อสร้างทุกคำ ภาพแปลประโยคแสดงวิธีเก็บ state ของ Encoder ทุกตำแหน่ง แล้วให้สถานะของ Decoder ปัจจุบันเป็นเงื่อนไขในการให้คะแนน $e_{t,j}$ กับ encoder state $h_j$ คำนวณ $\alpha_{t,j}=\operatorname{softmax}_j(e_{t,j})$ และ $c_t=\sum_j\alpha_{t,j}h_j$ ดังนั้น context $c_t$ เปลี่ยนตามคำที่กำลังสร้าง สไลด์ยก Bahdanau แบบ additive และ Luong แบบ multiplicative เป็นทางเลือกของการคำนวณ score

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 1–10

<a name="attention"></a>

## Scaled Dot-Product และ Multi-head Attention

ถ้ามี $T_q$ Queries และ $T_k$ Keys ใน head หนึ่ง ให้ $Q$ มี shape $(T_q,d_k)$, $K$ มี $(T_k,d_k)$ และ $V$ มี $(T_k,d_v)$ ตารางคะแนน $QK^\top$ มีหนึ่งแถวต่อ Query และหนึ่งคอลัมน์ต่อ Key

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

Softmax ทำข้าม Keys ต่อแถว $M$ แทน additive mask โดยตำแหน่งที่กันออกได้คะแนนต่ำมากก่อน Softmax ผลลัพธ์มีขนาด $(T_q,d_v)$ คือ Value ผสมหนึ่งชุดต่อ Query หลาย Heads ใช้ projections แยกแล้ว concatenate และฉายกลับให้มิติตรงกับบล็อกถัดไป

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** เมื่อคะแนนหลัง scale ของ Query หนึ่งคือ $(0,0)$ และ Values คือ $(2,0),(0,4)$ น้ำหนักเป็น $(0.5,0.5)$ และผลคือ $(1,2)$ ถ้า mask กัน Key ตัวที่สอง น้ำหนักของตัวแรกเป็น 1 ผลจึงเป็น $(2,0)$

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 11–18

## Transformer Block, Position และ KV Cache

Encoder ใช้ Self-Attention บน input และ Feed-forward ต่อ Token พร้อม residual และ normalization ในแผนภาพ Decoder ใช้ causal mask ใน Self-Attention และ Cross-Attention ที่ Query มาจาก Decoder ส่วน Key/Value มาจาก Encoder ตำแหน่งในประโยคจึงต้องเข้ามาผ่าน positional information สไลด์แสดง sinusoidal encoding แล้วเปรียบกับ RoPE ที่หมุนคู่มิติของ Q/K เพื่อให้มุมสัมพัทธ์มีผลต่อ dot product

Self-Attention เต็มมีตารางขนาด T×T และมีต้นทุนสูงเมื่อลำดับยาว ใน autoregressive Inference, KV Cache เก็บ Key/Value ของ Tokens ก่อนหน้าเพื่อใช้ซ้ำเมื่อมี Query ใหม่ การเก็บนี้แลกด้วยหน่วยความจำที่โตตามบริบท ส่วน local/window Attention ลดตำแหน่งที่ต้องเทียบในบาง layers

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 11–20, 35, 39

## BERT, Decoder-only และการปรับโมเดล

BERT ใช้ Transformer Encoder สองทิศเพื่อสร้าง contextual representation ต้นฉบับอธิบาย Masked Language Modeling และ Next Sentence Prediction แล้วแสดงการเพิ่มหัวงานปลายทาง WangchanBERTa ทำให้เห็นว่าภาษาและ tokenizer ต้องเข้ากับ pretraining ส่วน decoder-only ใช้บริบทด้านซ้ายในการสร้าง token ต่อเนื่อง และ encoder–decoder เหมาะกับตัวอย่างที่ส่ง input หนึ่งชุดแล้วสร้าง output อีกชุด Lab เปรียบเทียบแบบ train-from-scratch กับ Pretrained BERT ที่ตรึงหรือเปิด layers ต่างกัน

สไลด์ DeepSeek และ GPT-OSS ยกการออกแบบด้าน MoE, KV memory, Multi-Token Prediction, GQA และ local/global attention เป็นกรณีศึกษาที่แก้ข้อจำกัดคนละด้าน PEFT จำกัดจำนวนพารามิเตอร์ที่ปรับ เช่น LoRA เก็บ backbone เดิมและฝึกผลคูณ low-rank ของเมทริกซ์สองตัว QLoRA ลด precision ของฐานเพื่อประหยัดหน่วยความจำ ส่วน Prompt Tuning ฝึกเวกเตอร์เพิ่มที่ input ในขณะที่ backbone คงที่

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 21–44

## Transformer กับภาพและข้อมูลชนิดอื่น

ViT ตัดภาพเป็น patches แล้วแทนแต่ละ patch เป็น token เพื่อส่งให้ Encoder DETR ใช้ Backbone และ Transformer เพื่อทำนายชุดวัตถุ ส่วน Swin จัด attention เป็น local windows และเลื่อนขอบหน้าต่างเพื่อให้ข้อมูลข้ามกลุ่ม DeiT ใช้ Distillation Token, MViT ทำ representation หลายสเกล และ DINO กับ MAE แสดงการฝึกที่อาศัยข้อมูลภาพไม่ติดป้ายตามรูปแบบในสไลด์ VPT เพิ่ม learnable prompts ในงานภาพ และ Hiera กับ DINOv3 เป็นกรณีต่อยอด ต้นฉบับยังยก AlphaFold และ Boltz-1 ในภาพรวมของงานทำนายโครงสร้าง และ ConvNeXt เป็นตัวอย่างพัฒนาการฝั่ง CNN

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 45–60

## State Space Models และ Mamba

ภาพ SSM เขียน $h_t=Ah_{t-1}+Bx_t$ และ $y_t=Ch_t+Dx_t$ โดย $h$ เก็บสถานะย่อ, $x$ คือ input และ $y$ คือผล ณ timestep นั้น S4 เชื่อมมุมมอง recurrence กับ convolution ส่วน Mamba เพิ่มการเลือกว่าจะส่งผ่านหรือจดจำข้อมูลตาม input และใช้การจัดคำนวณแบบ scan สไลด์เปรียบเทียบ recurrent state ขนาดคงที่กับ Transformer ที่เข้าถึง token ก่อนหน้าได้โดยตรงแต่ใช้ KV cache โตขึ้น ค่าที่วัดได้ขึ้นกับ Architecture, งาน และ implementation ที่พิจารณา

> **อ้างอิง** [class08/08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 61–67

## ทบทวนและเชื่อมบท

เริ่มอ่านจากแหล่งที่มาของ Q/K/V จากนั้นถามว่า mask ตัดตำแหน่งไหน และยังต้องใช้ข้อมูลอะไรเมื่อความยาวเพิ่มขึ้น ตอนเปรียบเทียบ Lab ให้แยกผลจาก Architecture ออกจาก pretraining, tokenizer และขนาดโมเดล

## คำถามทบทวน

1. หาขนาด score และ output เมื่อ Q เป็น (3,8), K เป็น (5,8) และ V เป็น (5,4)
2. เหตุใด Transformer Lab ต้องกัน PAD ทั้งใน Attention และใน masked mean

[คำถามและเฉลย](../review/practice-and-solutions.md#class08) · [ศัพท์](../review/glossary.md) · [สมการ](../review/formula-reference.md)

## ที่มารายหัวข้อ

| หัวข้อ | ตำแหน่ง |
| --- | --- |
| จาก Thought Vector ไปสู่ Attention | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 1–10 |
| Scaled Dot-Product และ Multi-head Attention | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 11–18 |
| Transformer Block, Position และ KV Cache | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 11–20, 35, 39 |
| BERT, Decoder-only และการปรับโมเดล | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 21–44 |
| Transformer กับภาพและข้อมูลชนิดอื่น | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 45–60 |
| State Space Models และ Mamba | [08_attention_transformer_2handout_color.pdf](https://drive.google.com/file/d/1vtzVAzuxC18tJ9uskw_t9g4MzczAzXTC/view) · PDF หน้า 61–67 |

[← ก่อนหน้า](07-lstm-gru.md) | [Index](../INDEX.md) | [ถัดไป →](09-adversarial-attack-generative-ai.md)
