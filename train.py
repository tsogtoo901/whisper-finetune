#!/usr/bin/env python3
"""
Step 2 of the pipeline: fine-tune Whisper on the prepared dataset.

This is the standard, well-documented HuggingFace Seq2Seq fine-tuning
recipe for Whisper (community recipe), configured for Mongolian
transcription. Nothing exotic — per the spec, only two levers matter:

  * --learning_rate  (default 1e-5; too high causes "catastrophic
    forgetting" — the model gets worse at everything, loss may spike/NaN)
  * epochs — handled automatically: after every epoch the model is scored
    on the validation split, and training STOPS EARLY when validation WER
    stops improving (patience = 2 epochs). The best checkpoint, not the
    last one, is what gets saved. You do not tune epoch count by hand.

TWO MODES
---------
  Smoke test (cheap pipeline validation on whisper-small, ~30 min):
      python train.py --dataset_dir /workspace/prepared/dataset \
          --model_name openai/whisper-small \
          --output_dir /workspace/runs/smoke --smoke_test

  Real run (the quotable number, whisper-medium):
      python train.py --dataset_dir /workspace/prepared/dataset \
          --model_name openai/whisper-medium \
          --output_dir /workspace/runs/medium

The deliverable model lands in <output_dir>/final.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import jiwer
import torch
from datasets import load_from_disk
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from normalization import normalize_mn


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads audio features and token labels into batches (standard recipe)."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # If the tokenizer already prepended the decoder start token, strip it
        # (the model adds it itself during training).
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_name", default="openai/whisper-small")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--patience", type=int, default=2,
                    help="Stop after this many epochs without validation-WER improvement")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--smoke_test", action="store_true",
                    help="Tiny subset + 100 steps, just to prove the pipeline runs end to end")
    args = ap.parse_args()

    print(f"Loading processor + model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language="mongolian", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "mongolian"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False  # required with gradient checkpointing

    print(f"Loading dataset from {args.dataset_dir}")
    ds = load_from_disk(args.dataset_dir)
    train_ds, val_ds = ds["train"], ds["validation"]
    if args.smoke_test:
        train_ds = train_ds.select(range(min(256, len(train_ds))))
        val_ds = val_ds.select(range(min(64, len(val_ds))))
        print(f"SMOKE TEST: {len(train_ds)} train clips, {len(val_ds)} val clips")

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    print("Extracting audio features (cached after first run)...")
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                            num_proc=args.num_workers)
    val_ds = val_ds.map(prepare, remove_columns=val_ds.column_names,
                        num_proc=args.num_workers)

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        # SAME normalization as the final benchmark — imported, not copied.
        pairs = [(normalize_mn(r), normalize_mn(h))
                 for r, h in zip(label_str, pred_str)]
        pairs = [(r, h) for r, h in pairs if r]
        refs = [r for r, _ in pairs]
        hyps = [h for _, h in pairs]
        return {
            "wer": 100.0 * jiwer.wer(refs, hyps),
            "cer": 100.0 * jiwer.cer(refs, hyps),
        }

    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        report_to=["tensorboard"],
        dataloader_num_workers=args.num_workers,
    )
    if args.smoke_test:
        training_args = Seq2SeqTrainingArguments(
            **common, max_steps=100, warmup_steps=10,
            eval_strategy="steps", eval_steps=50,
            save_strategy="steps", save_steps=50,
        )
        callbacks = []
    else:
        training_args = Seq2SeqTrainingArguments(
            **common, num_train_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            eval_strategy="epoch", save_strategy="epoch",
        )
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nDONE. Best model saved to: {final_dir}")
    print("Next step: evaluation (see README, Phase 8).")


if __name__ == "__main__":
    main()
