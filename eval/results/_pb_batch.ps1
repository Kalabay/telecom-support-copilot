$env:HF_HOME="K:\dev\coursework\.hf_cache"; $env:TORCH_HOME="K:\dev\coursework\.caches\torch"; $env:PYTHONIOENCODING="utf-8"
$backends = @("vikhr_nemo12","tlite21","gigachat20_v15","qwen3moe","qwen3_14b")
foreach ($b in $backends) {
  $env:LLM_BACKEND=$b
  Write-Output "===== START $b $(Get-Date -Format HH:mm:ss) ====="
  & "K:\dev\coursework\backend\.venv\Scripts\python.exe" "K:\dev\coursework\eval\playbook_bo3_gen.py" 2>&1 | Select-String -NotMatch "Warning|warn|Batches|it/s|print_info|^ggml|^llama|CUDA|loaded meta"
  Write-Output "===== DONE $b $(Get-Date -Format HH:mm:ss) ====="
}
Write-Output "===== ВСЕ ПЛЕЙБУК-МОДЕЛИ ГОТОВЫ ====="
