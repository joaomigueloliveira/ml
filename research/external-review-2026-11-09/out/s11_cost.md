
## A: cost.json total 3.84 GPU-h; rows 54561; keys ['batch_size', 'cost', 'final_train_loss', 'grad_accum', 'gradient_checkpointing', 'group_by_length', 'max_length', 'name', 'train_runtime_s']
step 500: GPU-h 0.56, rows seen 8000, val exact 78.7, McNemar vs selected (3411) p=7.83e-09
step 1000: GPU-h 1.11, rows seen 16000, val exact 81.0, McNemar vs selected (3411) p=2.24e-05
step 1500: GPU-h 1.68, rows seen 24000, val exact 80.8, McNemar vs selected (3411) p=2.18e-05
step 2000: GPU-h 2.25, rows seen 32000, val exact 84.2, McNemar vs selected (3411) p=0.56
step 2500: GPU-h 2.80, rows seen 40000, val exact 83.7, McNemar vs selected (3411) p=0.0755
step 3000: GPU-h 3.37, rows seen 48000, val exact 84.4, McNemar vs selected (3411) p=0.453
step 3411: GPU-h 3.84, rows seen 54576, val exact 84.7, McNemar vs selected (3411) p=1

## B: cost.json total 5.11 GPU-h; rows 52424; keys ['batch_size', 'cost', 'final_train_loss', 'grad_accum', 'gradient_checkpointing', 'group_by_length', 'max_length', 'name', 'rows', 'teacher_gpu_hours', 'total_gpu_hours', 'train_runtime_s']
step 500: GPU-h 2.00, rows seen 8000, val exact 62.5, McNemar vs selected (1500) p=5.07e-11
step 1000: GPU-h 2.55, rows seen 16000, val exact 70.0, McNemar vs selected (1500) p=0.456
step 1500: GPU-h 3.12, rows seen 24000, val exact 70.8, McNemar vs selected (1500) p=1
step 2000: GPU-h 3.68, rows seen 32000, val exact 69.8, McNemar vs selected (1500) p=0.26
step 2500: GPU-h 4.24, rows seen 40000, val exact 70.0, McNemar vs selected (1500) p=0.374
step 3000: GPU-h 4.80, rows seen 48000, val exact 69.9, McNemar vs selected (1500) p=0.289
step 3277: GPU-h 5.11, rows seen 52432, val exact 70.1, McNemar vs selected (1500) p=0.427

## Bf: cost.json total 3.98 GPU-h; rows 37696; keys ['batch_size', 'cost', 'final_train_loss', 'grad_accum', 'gradient_checkpointing', 'group_by_length', 'max_length', 'name', 'rows', 'teacher_gpu_hours', 'total_gpu_hours', 'train_runtime_s']
step 500: GPU-h 1.97, rows seen 8000, val exact 73.1, McNemar vs selected (1000) p=0.00049
step 1000: GPU-h 2.50, rows seen 16000, val exact 76.6, McNemar vs selected (1000) p=1
step 1500: GPU-h 3.04, rows seen 24000, val exact 75.0, McNemar vs selected (1000) p=0.097
step 2000: GPU-h 3.59, rows seen 32000, val exact 76.3, McNemar vs selected (1000) p=0.815
step 2356: GPU-h 3.98, rows seen 37696, val exact 75.9, McNemar vs selected (1000) p=0.505

## D: cost.json total 15.38 GPU-h; rows 54561; keys ['batch_size', 'cost', 'engine', 'final_train_loss', 'grad_accum', 'name', 'total_gpu_hours', 'train_runtime_s']
step 500: GPU-h 2.22, rows seen 8000, val exact 70.0, McNemar vs selected (3000) p=0.621
step 1000: GPU-h 4.50, rows seen 16000, val exact 65.7, McNemar vs selected (3000) p=1.78e-07
step 1500: GPU-h 6.80, rows seen 24000, val exact 70.3, McNemar vs selected (3000) p=0.801
step 2000: GPU-h 9.08, rows seen 32000, val exact 69.1, McNemar vs selected (3000) p=0.0237
step 2500: GPU-h 11.34, rows seen 40000, val exact 69.9, McNemar vs selected (3000) p=0.21
step 3000: GPU-h 13.54, rows seen 48000, val exact 70.6, McNemar vs selected (3000) p=1
step 3410: GPU-h 15.38, rows seen 54560, val exact 70.6, McNemar vs selected (3000) p=1

## BD: cost.json total 4.23 GPU-h; rows 54561; keys ['batch_size', 'cost', 'engine', 'final_train_loss', 'grad_accum', 'name', 'total_gpu_hours', 'train_runtime_s']
step 100: GPU-h 0.41, rows seen 1600, val exact 67.7, McNemar vs selected (400) p=1.43e-05
step 200: GPU-h 0.82, rows seen 3200, val exact 69.0, McNemar vs selected (400) p=0.0385
step 300: GPU-h 1.23, rows seen 4800, val exact 68.9, McNemar vs selected (400) p=0.00592
step 400: GPU-h 1.65, rows seen 6400, val exact 70.4, McNemar vs selected (400) p=1
step 500: GPU-h 2.07, rows seen 8000, val exact 69.1, McNemar vs selected (400) p=0.0192
step 1000: GPU-h 4.23, rows seen 16000, val exact 68.4, McNemar vs selected (400) p=0.00779

## D step 500 vs step 3000
validation: only D500 48 / only D3000 54, p=0.621
test: only D500 76 / only D3000 106, p=0.0313
test-clean: only D500 50 / only D3000 89, p=0.00119
bfcl: only D500 31 / only D3000 69, p=0.000183

## Adapter check on BFCL: rows whose output differs from the raw 0.8B's (fc-eval reruns the first 32 without the adapter)
A: first 32 differ from saved raw run 18/32 (fc-eval's check: None); all 621/1000; raw-0.8B exact on first 32 28/32
B: first 32 differ from saved raw run 11/32 (fc-eval's check: None); all 528/1000; raw-0.8B exact on first 32 28/32
Bf: first 32 differ from saved raw run 7/32 (fc-eval's check: {'rows': 32, 'changed': 7, 'min_changed': 0.15}); all 545/1000; raw-0.8B exact on first 32 28/32
D: first 32 differ from saved raw run 7/32 (fc-eval's check: {'rows': 32, 'changed': 6, 'min_changed': 0.15}); all 515/1000; raw-0.8B exact on first 32 28/32
D500: first 32 differ from saved raw run 7/32 (fc-eval's check: {'rows': 32, 'changed': 6, 'min_changed': 0.15}); all 538/1000; raw-0.8B exact on first 32 28/32
BD: first 32 differ from saved raw run 7/32 (fc-eval's check: {'rows': 32, 'changed': 6, 'min_changed': 0.15}); all 509/1000; raw-0.8B exact on first 32 28/32
