# Rule-Based QA Generation with LoRA for Surgical Video Question Answering

This repository contains the implementation of our submission to the SurgVU 2025 Track 2.  
We fine-tuned a vision-language model using ~70k **rule-based generated QA pairs** derived from the provided annotations.  
LoRA (Low-Rank Adaptation) was used for efficient fine-tuning.

## Method Overview
1. **QA Generation**
   - Based on provided annotations, we generated ~70k QA pairs:
     - Instrument presence  
     - Organ presence  
     - Surgical phase (task)  
     - Purpose of instruments (rule-based mapping)

2. **Model**
   - Base model: InternVL3-2B  
   - Fine-tuning with **LoRA**  
   - Max 3 epochs (~28k steps)  
   - Training input: single randomly sampled frame per video segment  
   - Inference: central frame of the segment  

3. **Results**
   - Performance improved up to ~3,000 steps  
   - BLEU score degraded after ~28,000 steps due to overfitting  
   - Suggests need for more diverse QA generation (LLM-based)

---

## Training

We fine-tuned **InternVL3-2B** using [ms-swift](https://github.com/modelscope/ms-swift) with LoRA.  
Training data: ~70k rule-based QA pairs.

### Command

```bash
swift sft \
  --model OpenGVLab/InternVL3-2B \
  --use_hf true \
  --train_type lora \
  --lora_rank 64 --lora_alpha 16 --lora_dropout 0.05 \
  --dataset ./data/final_448_messages.jsonl \
  --columns '{"messages":"messages","images":"images"}' \
  --bf16 true --fp16 false \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --logging_steps 10 \
  --save_steps 1000 \
  --max_pixels 1003520 \
  --report_to tensorboard \
  --optim adamw_torch \
  --logging_nan_inf_filter false
```
