import os
import torch
import pytorch_lightning as pl
import albumentations as A
import rioxarray  # Nur importieren, NICHT benutzen! Registrierung reicht.
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from treecrowndelineation import TreeCrownDelineationModel
from treecrowndelineation.dataloading.in_memory_datamodule import InMemoryDataModule

# ===============================================================
# REPRODUZIERBARKEIT: Seed für alle Random-Operationen setzen
# HARMONISIERT: Prio 10 - Seed = 42 (identisch für alle 3 Programme)
# ===============================================================
SEED = 42
pl.seed_everything(SEED)
print(f"🌱 Seed gesetzt auf: {SEED} (harmonisiert)")

# ================= Pfade ===================
# HARMONISIERT: Prio 1 - Identische Input-Daten
rasters = "/home/abrax/Desktop/TCD_train_input/tiles"
masks = "/home/abrax/Desktop/TCD_train_input/masks"
outlines = "/home/abrax/Desktop/TCD_train_input/outlines"
dist = "/home/abrax/Desktop/TCD_train_input/dist_trafo"
logdir = "/home/abrax/Desktop/TCD_output/logs"
model_save_path = "/home/abrax/Desktop/TCD_output/models"
experiment_name = "Comparison_9params_TCD_final"  # Name für 9-Parameter-Vergleich

# ================ HARMONISIERTE HYPERPARAMETER ============
# HARMONISIERT: Prio 4 - Image Preprocessing (256×256, RGB)
arch = "Unet-resnet34"
width = 256  # ✓ 256×256 Tiles
batchsize = 4  # HARMONISIERT: Prio 7 - Batch Size = 4
in_channels = 3  # RGB (3 Kanäle)
max_epochs = 90  # HARMONISIERT: Prio 8 - 90 Epochen (wie gewünscht)
lr = 1e-4  # HARMONISIERT: Prio 6 (teilweise) - Learning Rate = 0.0001
training_split = 0.8  # HARMONISIERT: Prio 3 - 80-20 Split

# HARMONISIERT: Prio 12 (teilweise) - Validation alle 5 Epochen
val_check_interval = 1  # Validation alle 5 Epochen

# ========== Device-Auswahl/Strategy ==========
gpus = torch.cuda.device_count()
if gpus > 0:
    print(f"🚀 GPU verfügbar: {gpus} GPU(s)")
    trainer = pl.Trainer(gpus=gpus)
else:
    print("⚠️  Kein GPU verfügbar, nutze CPU")
    trainer = pl.Trainer()

# ============== Logging & Checkpoints ========
model_name = f"{arch}_9params_e{max_epochs}_lr{lr}_w{width}_bs{batchsize}_seed{SEED}"

# TensorBoard Logger (für interaktive Visualisierung)
tb_logger = TensorBoardLogger(
    save_dir=logdir,
    name=experiment_name,
    version=model_name,
    default_hp_metric=False
)

# CSV Logger (WICHTIG: für statistische Analyse & Vergleich!)
csv_logger = CSVLogger(
    save_dir=logdir,
    name=experiment_name,
    version=model_name
)

# Beide Logger kombinieren
logger = [tb_logger, csv_logger]

# Model Checkpoint: Speichere bestes Modell basierend auf val_loss
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(os.path.abspath(model_save_path), experiment_name),
    filename=model_name + "-{epoch:02d}-{val/loss:.4f}",
    monitor="val_loss",
    save_last=True,
    save_top_k=3,  # Speichere Top 3 Modelle
    mode="min",
    verbose=True
)

# Early Stopping: Stoppt bei Plateau
# HARMONISIERT: Prio 12 (teilweise) - Konsistente Validierungsstrategie
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=100,  # Warte 15 Epochen ohne Verbesserung (3×5 = 15)
    mode="min",
    verbose=True,
    min_delta=0.0001  # Minimale Änderung, die als Verbesserung gilt
)

# Learning Rate Monitor: Tracke LR-Änderungen
lr_monitor = LearningRateMonitor(logging_interval="epoch")

# Alle Callbacks
callbacks = [
    checkpoint_callback,
    lr_monitor,
    early_stop_callback
]

# ============== AUGMENTIERUNG (HARMONISIERT) ================
# HARMONISIERT: Prio 9 (teilweise) - Basis-Augmentierungen
# Angepasst an DeepTrees/Detectree2: HorizontalFlip, VerticalFlip, RandomRotate90
train_augmentation = A.Compose([
    A.RandomCrop(width, width),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Keine weiteren komplexen Augs, um mit Detectree2 vergleichbar zu bleiben
])

val_augmentation = A.RandomCrop(width, width)

print(f"\n{'='*70}")
print(f"🔧 AUGMENTIERUNGEN (harmonisiert):")
print(f"{'='*70}")
print(f"  Training:")
print(f"    - RandomCrop({width}, {width})")
print(f"    - HorizontalFlip (p=0.5)")
print(f"    - VerticalFlip (p=0.5)")
print(f"    - RandomRotate90 (p=0.5)")
print(f"  Validation:")
print(f"    - RandomCrop({width}, {width})")
print(f"{'='*70}\n")

# =========== Datamodule ======================
data = InMemoryDataModule(
    rasters,
    (masks, outlines, dist),
    width=width,
    batchsize=batchsize,
    training_split=training_split,
    train_augmentation=train_augmentation,
    val_augmentation=val_augmentation,
    concatenate_ndvi=False,
    red=0,
    nir=None,
    dilate_second_target_band=2,
    rescale_ndvi=False
)

# =========== Model-Initialisierung ===========
# HINWEIS: Optimizer ist Adam (TCD-Standard, nicht änderbar)
# HARMONISIERT: Prio 6 (teilweise) - lr=1e-4
model = TreeCrownDelineationModel(
    in_channels=in_channels,
    lr=lr
)

print(f"\n{'='*70}")
print(f"🎯 TRAINING KONFIGURATION (9/11 PARAMETER HARMONISIERT)")
print(f"{'='*70}")
print(f"  [✓] Prio 1:  Input-Daten      = Identische TIFs + SHP")
print(f"  [✗] Prio 2:  Label-Modus      = Semantic Seg. (nicht änderbar)")
print(f"  [✓] Prio 3:  Train-Test-Split = {training_split*100:.0f}% / {(1-training_split)*100:.0f}% (seed={SEED})")
print(f"  [✓] Prio 4:  Preprocessing    = {width}×{width} px, RGB (3ch)")
print(f"  [✗] Prio 5:  Loss             = TCD-intern (nicht änderbar)")
print(f"  [~] Prio 6:  Optimizer & LR   = Adam (Standard), lr={lr}")
print(f"  [✓] Prio 7:  Batch Size       = {batchsize}")
print(f"  [✓] Prio 8:  Epochen          = {max_epochs}")
print(f"  [~] Prio 9:  Augmentierungen  = Basis-Augs (harmonisiert)")
print(f"  [✓] Prio 10: Seed             = {SEED}")
#print(f"  [-] Prio 11: Device/Setup     = {accelerator} (nicht relevant)")
print(f"  [~] Prio 12: Validierung      = Alle {val_check_interval} Epochen")
print(f"{'='*70}")
print(f"  → 6 vollständig harmonisiert [✓]")
print(f"  → 3 teilweise harmonisiert [~]")
print(f"  → 2 nicht änderbar [✗]")
print(f"  → 1 nicht relevant [-]")
print(f"{'='*70}\n")

# ========== Trainer-Objekt ===================
gpus = torch.cuda.device_count()
if gpus > 0:
    trainer = Trainer(
        gpus=gpus,
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        check_val_every_n_epoch=val_check_interval,
        log_every_n_steps=10,
        deterministic=True,
        precision=32,
        gradient_clip_val=1.0,
    )
    print(f"🚀 GPU verfügbar: {gpus} GPU(s)")
else:
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        check_val_every_n_epoch=val_check_interval,
        log_every_n_steps=10,
        deterministic=True,
        precision=32,
        gradient_clip_val=1.0,
    )
    print("⚠️  Kein GPU verfügbar, nutze CPU")


# ========== Training starten ==================
trainer.fit(model, data)

# ========== Training abgeschlossen ============
print(f"\n{'='*70}")
print("✅ TRAINING ABGESCHLOSSEN!")
print(f"{'='*70}")
print(f"📊 Beste Modelle gespeichert unter:")
print(f"   {checkpoint_callback.dirpath}")
print(f"\n📈 Logs verfügbar unter:")
print(f"   TensorBoard: {tb_logger.log_dir}")
print(f"   CSV:         {csv_logger.log_dir}")
print(f"\n💡 TensorBoard starten mit:")
print(f"   tensorboard --logdir {logdir}")
print(f"{'='*70}\n")

# ========== Finale Metriken ausgeben ==========
if trainer.callback_metrics:
    print("📊 FINALE METRIKEN:")
    print(f"{'='*70}")
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key:30s}: {value.item():.6f}")
    print(f"{'='*70}\n")

# ========== CSV-Metriken Info ==================
csv_metrics_file = os.path.join(csv_logger.log_dir, "metrics.csv")
print(f"📁 WICHTIG: Metriken für statistischen Vergleich:")
print(f"{'='*70}")
print(f"   Datei: {csv_metrics_file}")
print(f"\n   Enthält pro Epoche (alle {val_check_interval} Epochen):")
print(f"   ✓ epoch         - Epochen-Nummer")
print(f"   ✓ train/loss    - Training Loss")
print(f"   ✓ val/loss      - Validation Loss (Hauptmetrik)")
print(f"   ✓ lr            - Learning Rate")
print(f"   ✓ weitere TCD-Metriken (z.B. IoU, wenn verfügbar)")
print(f"\n   → Diese CSV direkt für Vergleich mit Detectree2 & DeepTrees nutzen!")
print(f"{'='*70}\n")

print(f"✨ TCD Training mit 9/11 harmonisierten Parametern abgeschlossen!")
print(f"   Nächste Schritte:")
print(f"   1. Detectree2 mit gleichen Parametern trainieren")
print(f"   2. DeepTrees mit gleichen Parametern trainieren")
print(f"   3. CSV-Metriken aller 3 Programme vergleichen")
print(f"{'='*70}\n")

# ========== TorchScript Export (.pt) ==========
print(f"\n{'='*70}")
print("🔄 EXPORTIERE TORCHSCRIPT-MODELL (.pt)...")
print(f"{'='*70}")

# Modell auf CPU laden
model.to("cpu")

# Input-Tensor für Tracing erstellen
example_input = torch.rand(1, in_channels, width, width, dtype=torch.float32)

# TorchScript-Export Pfad (im übergeordneten Experiment-Ordner)
torchscript_path = os.path.join(
    os.path.abspath(model_save_path),
    experiment_name,
    model_name + "_jitted.pt"
)

# Export durchführen
model.to_torchscript(
    torchscript_path,
    method="trace",
    example_inputs=example_input
)

print(f"✅ TorchScript-Modell gespeichert unter:")
print(f"   {torchscript_path}")
print(f"{'='*70}\n")


