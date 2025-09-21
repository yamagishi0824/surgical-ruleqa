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

## 📂 Repository Structure
(Currently under preparation. Full training code will be released soon.)
