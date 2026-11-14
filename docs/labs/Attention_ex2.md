# Lab Attention_ex2

[Lab Index](README.md) · [Lecture 8](../lectures/08-attention-transformer.md) · [Index](../INDEX.md)

## เป้าหมาย

ศึกษาการ Freeze และ Fine-tune BERT-Small/BERT-Base บน IMDB เทียบกับ LSTM baseline

ต้นฉบับ [class08/code/Attention_ex2.ipynb](https://drive.google.com/file/d/190bmQVvigoPy-ey9Pt55Z6tlbboVxnFA/view) · ตรวจ 2026-11-14 · นับ cell จาก 1 รวม Markdown และ Code

## Environment และ dependencies

ใช้ PyTorch, NumPy, Matplotlib, Keras, Hugging Face transformers และ datasets รวมถึง torchinfo โหลด Dataset, tokenizers และ pretrained weights ผ่าน network โค้ดวัด peak CUDA memory โดยตรง จึงควรตรวจเส้นทางนี้หากรันบน CPU/MPS

## ข้อมูลสองเส้นทางและ Mask

LSTM ใช้ Keras IMDB ที่เป็น token IDs และ lengths ส่วน BERT อ่านข้อความ IMDB จาก Hugging Face แล้ว tokenize ด้วย model ID ที่ตรงกับ backbone ได้ input_ids และ attention_mask (B,T) โดย T=300 Labels เป็น float แล้วจัด (B,1) ก่อนคำนวณ Loss

> **อ้างอิง** [class08/code/Attention_ex2.ipynb](https://drive.google.com/file/d/190bmQVvigoPy-ey9Pt55Z6tlbboVxnFA/view) · cells 11–26

## BERT Head และการเปิด Layers

โหลด BertModel โดยไม่ใช้ pooling layer เริ่มตรึง BERT แล้วเลือกเปิด encoder blocks ท้ายและ embeddings ตาม config masked mean รวม last_hidden_state เฉพาะ token จริง ให้เวกเตอร์ (B,H) ก่อน Dropout, Linear และ Sigmoid สำหรับแบบที่เปิด Backbone ใช้ AdamW สองกลุ่มและ backbone LR แยกจาก head LR

> **อ้างอิง** [class08/code/Attention_ex2.ipynb](https://drive.google.com/file/d/190bmQVvigoPy-ey9Pt55Z6tlbboVxnFA/view) · cells 27–30, 35–43

## ฝึกและอ่านเปรียบเทียบ

ฝึก LSTM scratch หนึ่งแบบและ BERT หกแบบ รวม Frozen, Top layers, Encoder Fine-tuned และ Small ที่เปิด Embeddings ใช้ BCELoss และ threshold 0.5 Cells 15 และ 36 ใช้ test split ของแต่ละ Dataset เป็น validation ระหว่างฝึก ผลในกราฟจึงเป็นค่าจากชุดที่ใช้ติดตามการฝึก และยังมีความต่างของ tokenization กับ pretraining ระหว่าง baseline กับ BERT

> **อ้างอิง** [class08/code/Attention_ex2.ipynb](https://drive.google.com/file/d/190bmQVvigoPy-ey9Pt55Z6tlbboVxnFA/view) · cells 30–52

## อ่านคำสั่งสำคัญ

ตัวอย่างจาก cell 28

```python
out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
```

BERT ได้ทั้ง token IDs และตำแหน่งที่ใช้ได้ จากนั้น custom head เฉลี่ย Features โดยใช้ mask เดิม

## ผลลัพธ์และการตีความ

**Output saved in source notebook** Cells 38–43 มี logs และ widget outputs ของ BERT runs Cells 48, 50 และ 52 เก็บกราฟ Loss, Accuracy และการใช้ทรัพยากร


**Output reproduced by Codex** ไม่มีการรัน Notebook ผลที่ระบุเป็น output ที่บันทึกอยู่ในต้นฉบับ ส่วน shape หรือพฤติกรรมที่อธิบายจากโค้ดเป็น **Expected behavior based on reading code**

## แบบฝึกและการตรวจข้อผิดพลาด

**Illustrative Example (ผู้เรียบเรียงสร้างขึ้น)** วาดสอง optimizer groups สำหรับ BERT-Base Top-3 แล้วไล่ว่า parameter ไหนยัง frozen หากเกิด error เกี่ยวกับ peak memory ให้ตรวจ calls ของ torch.cuda ก่อนวินิจฉัยว่าข้อมูลหรือ BERT ผิด

[Lab Index](README.md) · [Lecture 8](../lectures/08-attention-transformer.md)
