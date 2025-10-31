# --- Monitor-InternetAndDocker.ps1 ---

# CONFIGURATION
$containerName = "your_container_name"  # Change this to your container name
$checkInterval = 30                     # Time in seconds between checks
$testUrl = "https://www.google.com"     # Reliable test URL

function Test-InternetConnection {
    try {
        $response = Invoke-WebRequest -Uri $testUrl -UseBasicParsing -TimeoutSec 5
        return $true
    } catch {
        return $false
    }
}

function Restart-DockerContainer {
    param([string]$container)

    Write-Host "Attempting to restart Docker container: $container"

    # Check if container is running
    $isRunning = docker ps --format "{{.Names}}" | Where-Object { $_ -eq $container }

    if ($isRunning) {
        Write-Host "Stopping container $container..."
        docker stop $container | Out-Null
    } else {
        Write-Host "Container $container is not running."
    }

    Write-Host "Starting container $container..."
    docker start $container | Out-Null

    Write-Host "Container restart complete ✅"
}

Write-Host "Starting Internet Monitor Service..."
Write-Host "Monitoring internet connection every $checkInterval seconds."
Write-Host "Docker container: $containerName"
Write-Host "-------------------------------------------"

while ($true) {
    $isOnline = Test-InternetConnection

    if (-not $isOnline) {
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Internet down ❌"
        Restart-DockerContainer -container $containerName
    } else {
        Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Internet OK ✅"
    }

    Start-Sleep -Seconds $checkInterval
}
