$env:HF_HOME="K:\dev\coursework\.hf_cache"; $env:TORCH_HOME="K:\dev\coursework\.caches\torch"; $env:PYTHONIOENCODING="utf-8"
$backends = @("mistral24","qwen3_14b","tlite21","vikhr_nemo12","gigachat20_v15","qwen3moe","ruadapt32","gemma27")
foreach ($b in $backends) {
  $env:LLM_BACKEND=$b
  Write-Output "===== START $b $(Get-Date -Format HH:mm:ss) ====="
  & "K:\dev\coursework\backend\.venv\Scripts\python.exe" "K:\dev\coursework\eval\bo3_model_gen.py" 2>&1 | Select-String -NotMatch "Warning|warn|Batches|it/s|print_info|^ggml|^llama|CUDA|loaded meta" 
  Write-Output "===== DONE $b $(Get-Date -Format HH:mm:ss) ====="
}
Write-Output "===== ВСЕ МОДЕЛИ ГОТОВЫ ====="
