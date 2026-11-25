param(
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param([string[]]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

# Ensure we run from repository root
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Test-Path ".git")) {
    Write-Error "Current directory is not a git repository. Run 'git init' first."
    exit 1
}

$Branch = (git branch --show-current 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($Branch)) {
    $Branch = "main"
}

# Check if origin remote is configured
$null = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] 尚未配置 GitHub 远程仓库 (origin)。" -ForegroundColor Yellow
    Write-Host "请先运行：" -ForegroundColor Cyan
    Write-Host "  git remote add origin https://github.com/<你的用户名>/<仓库名>.git"
    Write-Host "  .\scripts\sync.ps1 `"init: first push`""
    exit 0
}

# Stage & commit local changes if any
$Status = git status --porcelain
if ($Status) {
    $GitName = (git config user.name 2>$null)
    $GitEmail = (git config user.email 2>$null)
    if ([string]::IsNullOrWhiteSpace($GitName) -or [string]::IsNullOrWhiteSpace($GitEmail)) {
        Write-Host "[ERROR] Git 尚未配置用户名或邮箱，无法提交。" -ForegroundColor Red
        Write-Host "请先执行以下两条命令（只需配置一次）：" -ForegroundColor Yellow
        Write-Host "  git config --global user.name `"luyiming123`""
        Write-Host "  git config --global user.email `"你的GitHub邮箱`""
        exit 1
    }

    if ([string]::IsNullOrWhiteSpace($Message)) {
        $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $Message = "sync: update code and notes ($Timestamp)"
    }
    Write-Host "[1/3] 提交本地改动: $Message" -ForegroundColor Cyan
    Invoke-Git @("add", "-A")
    Invoke-Git @("commit", "-m", $Message)
} else {
    Write-Host "[1/3] 本地没有新的未提交改动。" -ForegroundColor Gray
}

# Pull remote updates (e.g., results pushed from Colab) if remote branch exists
$RemoteBranchExists = git ls-remote --heads origin $Branch
if ($LASTEXITCODE -ne 0) {
    throw "无法连接到远程仓库 origin，请检查网络或代理设置。"
}

if ($RemoteBranchExists) {
    Write-Host "[2/3] 拉取远端更新 (origin/$Branch)..." -ForegroundColor Cyan
    Invoke-Git @("pull", "--rebase", "origin", $Branch)
} else {
    Write-Host "[2/3] 远端分支 $Branch 尚不存在，准备首次推送..." -ForegroundColor Gray
}

Write-Host "[3/3] 推送到 GitHub (origin/$Branch)..." -ForegroundColor Cyan
Invoke-Git @("push", "-u", "origin", $Branch)

Write-Host "[DONE] 同步完成！" -ForegroundColor Green
