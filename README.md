
# 🧠 HysoLLM – Makine Öğrenimi & LLM Eğitim Altyapısı

HysoLLM, makine öğrenimi ve özellikle büyük dil modeli (LLM) projelerinde **eğitim süreçlerini düzenlemek, deneyleri kaydetmek, yeniden üretilebilir hale getirmek ve modüler bir mimari üzerine oturtmak** amacıyla geliştirilmiş hafif bir framework’tür.

Bu altyapı, modern ML projelerinde ihtiyaç duyulan temel bileşenleri tek çatı altında toplar:

- ⚙️ Modüler model mimarileri  
- 🔤 Tokenizer sistemi (BPE + Simple)  
- 📦 Deney yönetimi (run paths, manifest, logger, checkpoints, seed)  
- ⚡ Eğitim döngüsü (trainer, callback sistemi)  
- 🧾 Config yönetimi (YAML/JSON + CLI override)  
- 📁 Harici config dosyaları (configs/)  

---
## **🚀 Kullanım Örneği**

- Aşağıda HysoLLM bileşenlerinin birlikte nasıl çalıştığını gösteren gerçek bir örnek bulunmaktadır.
Amaç:  
**Model kodunu eğitim mekanizmalarından tamamen ayırarak**, farklı projelerde yeniden kullanılabilir, düzenli ve sürdürülebilir bir yapı oluşturmak.
```python
import sys
sys.path.append(r"C:\Users\hdgn5\OneDrive\Masaüstü\hysollm")

import json
import torch

# ======================================
# Hyso Core (config + storage)
# ======================================
from hyso.core.config import load_config_with_overrides
from hyso.core.storage import (
    RunPathFactory,
    get_logger,
    Manifest,
    save_manifest,
    set_global_seed,
    CheckpointConfig,
    CheckpointManager,
)

# ======================================
# Hyso Train & Tokenizer
# ======================================
from hyso.core.train.trainer import HysoTrainer
from hyso.core.train.metrics import LLMetrics
from hyso.core.train.callbacks import HysoCallbacks
from hyso.core.train.dataloader import build_dataloader   # <-- KÜTÜPHANEDE VAR
from hyso.core.tokenizer.bpe_tokenizer import HysoBPETokenizer

# ======================================
# Hyso Model
# ======================================
import hyso

# =====================================================================
# 1) CONFIG YÜKLE (YAML dosyasından + CLI override desteği)
# =====================================================================

cfg = load_config_with_overrides(
    path="configs/base.yaml",
    override_pairs=sys.argv[1:],   # Örn: training.lr=0.0001
)

# =====================================================================
# 2) STORAGE YAPISI KUR (RunPaths, Logger, Seed, Manifest)
# =====================================================================

factory = RunPathFactory.from_root("runs")
run_paths = factory.create()

logger = get_logger("train", log_dir=run_paths.logs_dir)
logger.info(f"Yeni run başlatıldı: {run_paths.run_id}")

set_global_seed(cfg.training.seed)

# =====================================================================
# 3) VERİSETİNİN NEREDE DURMASI GEREKİR?
# =====================================================================
"""
Gerçek ve profesyonel dizin yapısı:

project/
   data/
      train.json
      val.json
      test.json

Dosya formatı:
[
  {"src": "merhaba dünya", "tgt": "hello world"},
  {"src": "nasılsın", "tgt": "how are you"},
  ...
]

Bu format seq2seq için *standarttır* ve HysoTrainer'ın collate yapısıyla
doğrudan uyumludur.
"""

with open("data/train.json", "r", encoding="utf-8") as f:
    train_records = json.load(f)

with open("data/val.json", "r", encoding="utf-8") as f:
    val_records = json.load(f)

logger.info(f"Verisetleri yüklendi. Train: {len(train_records)}, Val: {len(val_records)}")

# =====================================================================
# 4) TOKENIZER TRAIN 
# =====================================================================

corpus = [item["src"] for item in train_records] + \
         [item["tgt"] for item in train_records]

tok = HysoBPETokenizer(
    lowercase=False,
    normalize="NFKC",
    cache_size=10000,
)

tok.fit(
    corpus=corpus,
    target_vocab_size=cfg.tokenizer.vocab_size,
    min_pair_freq=2,
)

logger.info(f"Tokenizer hazır. Vocab size: {tok.vocab_size}")

# =====================================================================
# 5) DATALOADER — KÜTÜPHANEDE KENDİ BUILD_DATALOADER VAR
# =====================================================================

train_loader = build_dataloader(
    dataset=train_records,     # list[dict{src,tgt}]
    mode="seq2seq",
    tokenizer=tok,
    batch_size=cfg.training.batch_size,
    max_src_len=cfg.data.max_src_len,
    max_tgt_len=cfg.data.max_tgt_len,
    shuffle=True,
)

val_loader = build_dataloader(
    dataset=val_records,
    mode="seq2seq",
    tokenizer=tok,
    batch_size=cfg.training.batch_size,
    max_src_len=cfg.data.max_src_len,
    max_tgt_len=cfg.data.max_tgt_len,
    shuffle=False,
)

# =====================================================================
# 6) MODEL OLUŞTUR
# =====================================================================

model = hyso.HysoLLM(
    vocab_src=tok.vocab_size,
    vocab_tgt=tok.vocab_size,
    d_model=cfg.model.dim,
    n_heads=cfg.model.heads,
).to(cfg.training.device)

logger.info("Model oluşturuldu.")

# =====================================================================
# 7) METRICS + CALLBACKS + CHECKPOINTMANAGER
# =====================================================================

metrics = LLMetrics(
    bleu=False,
    perplexity=True,
    token_accuracy=True,
    token_ignore_index=-100,
)

callbacks = HysoCallbacks.default(
    total_steps=len(train_loader) * cfg.training.epochs,
    save_dir=str(run_paths.checkpoints_dir),
)

ckpt_cfg = CheckpointConfig.from_dir(
    directory=run_paths.checkpoints_dir,
    best_metric_name="val_loss",
    best_mode="min",
)

checkpoint_manager = CheckpointManager(ckpt_cfg)

# =====================================================================
# 8) TRAINER — KÜTÜPHANE TAM ENTEGRE
# =====================================================================

trainer = HysoTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    tokenizer=tok,
    epochs=cfg.training.epochs,
    lr=cfg.training.lr,
    optimizer=cfg.training.optimizer,
    scheduler=cfg.training.scheduler,
    warmup_ratio=cfg.training.warmup_ratio,
    ignore_index=-100,
    callbacks=callbacks,
    metrics=metrics,
    logger=logger,
    run_paths=run_paths,
    checkpoint_manager=checkpoint_manager,
    config=cfg,
)

# =====================================================================
# 9) EĞİTİM BAŞLAT
# =====================================================================

result = trainer.fit()
logger.info("Eğitim tamamlandı.")


```
---

# 📁 Proje Yapısı
Her klasör aşağıda açıklanmaktadır.
```bash
core/
  models/
  tokenizer/
  storage/
  train/
  config/
  configs/
```

---

# 🔤 1. Tokenizer Modülü (`hyso/core/tokenizer/`)

Tokenizer yapısı metni modele uygun ID dizilerine dönüştürür. HysoLLM iki tokenizer içerir:

---

### **• HysoBPETokenizer (BPE)**  
Byte Pair Encoding tabanlı tokenizer.

**Yöntemler & yetenekler:**
- BPE merge kurallarına dayalı alt birim üretimi  
- Özel token ID yönetimi  
- Batch encode/decode  
- Seq2Seq encoder ve decoder için ayrı encode fonksiyonları  
- UTF-8 byte-level işleme desteği  

**Kullanılan Teknolojiler:**
- Python  
- PyTorch (tensor dönüşümleri)  
- Unicode processing  
- BPE algoritması  

---

### **• HysoSimpleTokenizer (Word-level)**  
Kelime temelli sade tokenizer.

**Yöntemler:**
- Temel whitespace tokenizasyonu  
- Kelime frekansına göre vocabulary oluşturma  
- Batch encode/decode  
- Mask üretimi  

**Kullanılan Teknolojiler:**
- Python  
- PyTorch  
- Temel NLP tokenizasyon teknikleri  

---

# 🧩 2. Model Modülü (`hyso/core/models/`)

Bu klasör HysoLLM model mimarilerini içerir. Tüm modeller modüler ve genişletilebilir yapıdadır.

---

### **Model mimarileri:**
- **Encoder-Decoder Transformer**  
- **Encoder-only LLM** (BERT/LLM tarzı)  
- **Decoder-only Model** (GPT tarzı)

---

### **Her modelin sunduğu yöntemler:**
- `forward()`  
- `generate()` (decoder-only)  
- `encode()` / `decode()` (enc-dec yapıları)  
- RoPE pozisyonel embedding  
- Attention katmanları  
- DropPath, LayerNorm, RMSNorm gibi modern bileşenler  

**Kullanılan Teknolojiler:**
- PyTorch (nn.Module)  
- Multi-Head Attention  
- Rotary Positional Embedding (RoPE)  
- SwiGLU / GeLU aktivasyonları  
- Residual / PreNorm mimarileri  

---

# 📦 3. Storage Modülü (`hyso/core/storage/`)

Bu modül HysoLLM’in gerçek “deney yönetimi” çekirdeğidir.
Klasör yapısını otomatik oluşturur.

**Yöntemler:**  
- `create()`  
- `run_id` üretimi  
- klasör kurulumu  

## **• Logger**  
Hem konsola hem dosyaya temiz log yazan yapılandırılabilir logger.

**Yöntemler:**  
- `get_logger(name, log_dir)`  
- Formatlama (timestamp, level, name, message)  
- Rotating log desteği  


## **• Manifest**  
Her eğitimin kimlik kartıdır.

İçerik:  
- model bilgisi  
- eğitim bilgisi  
- veri seti bilgisi  
- ortam (Python, OS, CUDA)  
- hyperparametreler  
- timestamp  

**Yöntemler:**  
- `Manifest.new()`  
- `save_manifest()`  
- `load_manifest()`  

## **• CheckpointManager**  
Eğitim sırasında modelleri güvenli şekilde kaydeder.

**Yöntemler:**
- `save(epoch, model, optimizer, scheduler)`  
- `load_latest()`  
- `load_best()`  
- `restore_objects()`  
- max_to_keep ile eski checkpoint silme  


## **• Seed Yönetimi**
Tüm ortamı deterministik hale getirir.

**Yöntemler:**  
- `set_global_seed(seed)`  

**Kullanılan Teknolojiler (tüm storage):**
- PyTorch  
- Python logging  
- JSON  
- UUID  
- Pathlib  
- Datetime  

---

# ⚡ 4. Train Modülü (`hyso/core/train/`)

Bu modül model eğitimi için gerekli tüm yapılara sahiptir.

### **İçerik:**

### **• Trainer**
Model, veri, optimizer, scheduler, storage ve config bileşenlerini bir araya getiren ana eğitim sınıfı.

**Yöntemler:**
- `train_epoch()`  
- `validate()`  
- `fit()`  
- Loss hesaplama  
- Metrics kaydetme  


### **• Callback Sistemi**
Eğitimin belirli aşamalarında çalışacak küçük kancalar.

**Yöntemler:**
- `on_train_start()`  
- `on_epoch_end()`  
- `on_step_end()`  
- CallbackList ile çoklu callback desteği  

### **• Metrics Logging**
Eğitim ve validasyon metriklerini CSV’e yazar.

**Kullanılan Teknolojiler:**
- PyTorch (dataset, dataloader, optim, scheduler)  
- Callback pattern  
- CSV logging  
- Checkpoint entegrasyonu  

# 🧾 5. Config Modülü (`hyso/core/config/`)

Config sistemi ayarları koddan ayırır ve tüm eğitim süreçlerini yapılandırılabilir hale getirir.


### **Yapılan işlemler:**
- YAML / JSON config dosyası yükleme  
- CLI override desteği  
  (`training.lr=0.0001 model.layers=12` gibi)  
- Deep merge  
- Config kaydetme / okuma  
- Attribute-style erişim:  
  `cfg.training.lr`  


### **Yöntemler:**
- `load_config(path)`  
- `parse_overrides(argv)`  
- `merge_config(base, override)`  
- `load_config_with_overrides()`  
- `save_config()`  

**Kullanılan Teknolojiler:**
- PyYAML  
- JSON  
- Python AST parsing  
- Recursive merge algoritması  

---

# 📁 6. Configs Klasörü (`configs/`)

Bu klasör train modülü için harici ayar dosyalarını içerir.

Örnek dosyalar:
- `base.yaml`
- `encoder_small.yaml`
- `encoder_large.yaml`
- `lr_sweep.yaml`

Bu sayede:

- Deney ayarları versiyonlanabilir,  
- Farklı modeller için hızlı switching yapılabilir,  
- Manifest + config birleşince tam reproducibility sağlanır.

**Kullanılan Teknolojiler:**  
- YAML  
- JSON  

---

# 🧩 Özet

HysoLLM yalnızca model mimarisi değil;  
**tam bir eğitim altyapısı, deney yönetim sistemi, config framework ve modüler ML yapı setidir.**

Bu repo:

- 🔤 Tokenizer sistemini  
- ⚙️ Model mimarisini  
- 📦 Storage ve deney yönetimini  
- ⚡ Eğitim altyapısını  
- 🧾 Config yönetimini  

birbirinden ayrılmış, temiz ve profesyonel şekilde organize eder.

--- 

# ✨ Lisans
MIT License



