param(
    [string]$HubUser = "beyond0814",
    [string]$Revision = "main",
    [switch]$Private
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$models = @(
    @{ Name = "ThinkOmni-Stage1-SFA"; Path = Join-Path $repoRoot "model\model_stage1_SFA" },
    @{ Name = "ThinkOmni-Stage2-SFA-AFA"; Path = Join-Path $repoRoot "model\model_stage2_SFA+AFA" },
    @{ Name = "ThinkOmni-Stage3-SFA-AFA-MFR"; Path = Join-Path $repoRoot "model\model_stage3_SFA+AFA+MFR" }
)

if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    throw "Hugging Face CLI not found. Install with: pip install -U huggingface_hub"
}

hf auth whoami

foreach ($model in $models) {
    if (-not (Test-Path $model.Path)) {
        throw "Missing model directory: $($model.Path)"
    }

    $repoId = "$HubUser/$($model.Name)"
    $args = @("upload-large-folder", $repoId, $model.Path, "--repo-type", "model", "--revision", $Revision, "--num-workers", "4")
    if ($Private) {
        $args += "--private"
    }

    Write-Host "Uploading $($model.Path) to $repoId ..."
    & hf @args
    if ($LASTEXITCODE -ne 0) {
        throw "Upload failed for $repoId"
    }
}

