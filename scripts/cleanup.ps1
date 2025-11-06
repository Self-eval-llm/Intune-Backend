# Cleanup script - Remove unnecessary files and folders
# Run this to free up disk space after successful training and deployment

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "CLEANUP: REMOVING UNNECESSARY FILES" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

$itemsToRemove = @()

# Large temporary folders
if (Test-Path "llama.cpp") {
    $itemsToRemove += @{Path="llama.cpp"; Type="Folder"; Reason="llama.cpp repo (can re-clone if needed)"; Size=(Get-ChildItem -Path "llama.cpp" -Recurse | Measure-Object -Property Length -Sum).Sum}
}

if (Test-Path "gemma-finetuned") {
    $itemsToRemove += @{Path="gemma-finetuned"; Type="Folder"; Reason="Training checkpoints (merged model already saved)"; Size=(Get-ChildItem -Path "gemma-finetuned" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum}
}

if (Test-Path "unsloth_compiled_cache") {
    $itemsToRemove += @{Path="unsloth_compiled_cache"; Type="Folder"; Reason="Unsloth compilation cache (will regenerate if needed)"; Size=(Get-ChildItem -Path "unsloth_compiled_cache" -Recurse | Measure-Object -Property Length -Sum).Sum}
}

# Intermediate GGUF (keep final one)
if (Test-Path "gemma-finetuned-f16.gguf") {
    $itemsToRemove += @{Path="gemma-finetuned-f16.gguf"; Type="File"; Reason="Intermediate F16 GGUF (final GGUF already created)"; Size=(Get-Item "gemma-finetuned-f16.gguf").Length}
}

# LoRA adapters only (merged model is better)
if (Test-Path "gemma-finetuned-lora") {
    $itemsToRemove += @{Path="gemma-finetuned-lora"; Type="Folder"; Reason="LoRA adapters only (merged model is complete)"; Size=(Get-ChildItem -Path "gemma-finetuned-lora" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum}
}

# Training data (keep originals)
if (Test-Path "train_dataset.jsonl") {
    $itemsToRemove += @{Path="train_dataset.jsonl"; Type="File"; Reason="Training data (can regenerate from Supabase)"; Size=(Get-Item "train_dataset.jsonl").Length}
}

if (Test-Path "val_dataset.jsonl") {
    $itemsToRemove += @{Path="val_dataset.jsonl"; Type="File"; Reason="Validation data (can regenerate from Supabase)"; Size=(Get-Item "val_dataset.jsonl").Length}
}

# Python cache
if (Test-Path "__pycache__") {
    $itemsToRemove += @{Path="__pycache__"; Type="Folder"; Reason="Python cache"; Size=(Get-ChildItem -Path "__pycache__" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum}
}

if (Test-Path "matrics/__pycache__") {
    $itemsToRemove += @{Path="matrics/__pycache__"; Type="Folder"; Reason="Python cache"; Size=(Get-ChildItem -Path "matrics/__pycache__" -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum}
}

# Show what will be removed
Write-Host "Items to be removed:" -ForegroundColor Yellow
Write-Host ""

$totalSize = 0
$index = 1
foreach ($item in $itemsToRemove) {
    $sizeMB = [math]::Round($item.Size / 1MB, 2)
    $totalSize += $item.Size
    Write-Host "[$index] " -NoNewline -ForegroundColor Cyan
    Write-Host "$($item.Path) " -NoNewline -ForegroundColor White
    Write-Host "($sizeMB MB)" -ForegroundColor Gray
    Write-Host "    Reason: $($item.Reason)" -ForegroundColor DarkGray
    Write-Host ""
    $index++
}

$totalSizeMB = [math]::Round($totalSize / 1MB, 2)
$totalSizeGB = [math]::Round($totalSize / 1GB, 2)

Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "Total space to free: " -NoNewline -ForegroundColor Yellow
Write-Host "$totalSizeMB MB ($totalSizeGB GB)" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

# Ask for confirmation
$confirmation = Read-Host "Do you want to proceed with cleanup? (yes/no)"

if ($confirmation -ne "yes") {
    Write-Host ""
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Starting cleanup..." -ForegroundColor Green
Write-Host ""

$removedCount = 0
$removedSize = 0

foreach ($item in $itemsToRemove) {
    if (Test-Path $item.Path) {
        try {
            if ($item.Type -eq "Folder") {
                Remove-Item -Path $item.Path -Recurse -Force -ErrorAction Stop
            } else {
                Remove-Item -Path $item.Path -Force -ErrorAction Stop
            }
            Write-Host "✓ Removed: $($item.Path)" -ForegroundColor Green
            $removedCount++
            $removedSize += $item.Size
        } catch {
            Write-Host "✗ Failed to remove: $($item.Path)" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor DarkRed
        }
    }
}

$removedSizeMB = [math]::Round($removedSize / 1MB, 2)
$removedSizeGB = [math]::Round($removedSize / 1GB, 2)

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "✅ CLEANUP COMPLETED!" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""
Write-Host "Removed: $removedCount items" -ForegroundColor Green
Write-Host "Freed space: $removedSizeMB MB ($removedSizeGB GB)" -ForegroundColor Green
Write-Host ""

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "WHAT'S KEPT (IMPORTANT FILES)" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""

$keptFiles = @(
    "gemma-finetuned-merged/          - Fine-tuned PyTorch model (for evaluation)",
    "gemma-finetuned.gguf             - GGUF model for Ollama",
    "Modelfile                        - Ollama configuration",
    ".env                             - Environment variables (Supabase credentials)",
    "finetune_gemma.py                - Training script",
    "evaluate_finetuned.py            - Evaluation script (PyTorch)",
    "evaluate_ollama.py               - Evaluation script (Ollama)",
    "llm_eval.py                      - Metrics computation",
    "output1.json                     - Original dataset",
    "*.py scripts                     - All Python scripts for future use"
)

foreach ($file in $keptFiles) {
    Write-Host "✓ $file" -ForegroundColor Gray
}

Write-Host ""
Write-Host ("=" * 80) -ForegroundColor Cyan
