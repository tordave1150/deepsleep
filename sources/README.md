# แหล่งเรียนและการตรวจย้อนกลับ

โฟลเดอร์ของวิชาที่ตรวจเมื่อ 2026-11-14 มี 11 โฟลเดอร์ Class และ 7 โฟลเดอร์ย่อย [Google Drive](https://drive.google.com/drive/folders/1SQNa3jVmmPyw1Qvaz0fT26JxgkxrBmzK)

## ใช้เลขอ้างอิงอย่างไร

- PDF ใช้เลขหน้าตาม viewer โดยนับปกเป็นหน้า 1 แต่ละหน้าของ handout มักมีสองสไลด์ เลขสไลด์ที่พิมพ์ไว้จึงต่างจากเลขหน้า
- Notebook ใช้ตำแหน่งตามลำดับของ cells ในไฟล์ JSON เริ่มที่ 1 นับ Markdown และ Code รวมกัน ค่า execution count ใน UI เป็นข้อมูลอีกชุดหนึ่ง
- [Manifest](manifest.json) บันทึก Drive ID, MIME type, checksum, จำนวนหน้า/เซลล์ และวันตรวจ [Coverage](coverage.md) ผูกแหล่งกับบทเรียนและ Lab
- Raw files และผลสกัดที่ใช้ตรวจอยู่ใต้ `.local_sources/` กับ `.local_audit/` ตามลำดับ โฟลเดอร์ทั้งสองถูก ignore โดย Git ลิงก์ต้นฉบับใช้สิทธิ์เดิมของ Drive ควรได้รับอนุญาตจากเจ้าของก่อนเผยแพร่ต้นฉบับต่อ

[สารบัญ](../docs/INDEX.md) · [สถานะ](../docs/STATUS.md)
