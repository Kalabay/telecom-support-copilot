$base = "$env:USERPROFILE\.claude\projects"
$files = Get-ChildItem $base -Recurse -Filter *.jsonl -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 50

$bashCounts = @{}
$mcpCounts = @{}

function Get-CommandKey {
    param([string]$cmd)
    if ([string]::IsNullOrWhiteSpace($cmd)) { return $null }
    $c = $cmd.Trim()
    # Take first segment before pipe/&&/||/;
    $c = ($c -split '\|\||&&|\|(?!\|)|;' | Select-Object -First 1).Trim()
    # Strip env-var prefixes: VAR=val cmd
    while ($c -match '^[A-Z_][A-Z0-9_]*=\S+\s+(.+)$') { $c = $Matches[1].Trim() }
    # Strip wrappers: sudo, timeout, command, env
    while ($c -match '^(sudo|timeout(\s+\d+\S*)?|command|env|nice|nohup)\s+(.+)$') {
        $c = $Matches[3].Trim()
    }
    # Now grab head + first arg if it looks like a subcommand (not a flag)
    $parts = $c -split '\s+', 3
    $head = $parts[0]
    if ($parts.Count -ge 2 -and $parts[1] -notmatch '^-' -and $parts[1] -match '^[a-z][a-z0-9:-]*$') {
        return "$head $($parts[1])"
    }
    return $head
}

foreach ($f in $files) {
    Get-Content -LiteralPath $f.FullName -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_ -notmatch '"tool_use"') { return }
        try {
            $obj = $_ | ConvertFrom-Json -ErrorAction Stop
        } catch { return }
        if ($obj.type -ne 'assistant') { return }
        $content = $obj.message.content
        if (-not $content) { return }
        foreach ($block in $content) {
            if ($block.type -ne 'tool_use') { continue }
            $name = $block.name
            if ($name -eq 'Bash') {
                $cmd = $block.input.command
                $key = Get-CommandKey $cmd
                if ($key) {
                    if ($bashCounts.ContainsKey($key)) { $bashCounts[$key]++ } else { $bashCounts[$key] = 1 }
                }
            } elseif ($name -like 'mcp__*') {
                if ($mcpCounts.ContainsKey($name)) { $mcpCounts[$name]++ } else { $mcpCounts[$name] = 1 }
            }
        }
    }
}

"=== BASH TOP 40 ==="
$bashCounts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 40 |
    ForEach-Object { "{0,5}  {1}" -f $_.Value, $_.Key }

"`n=== MCP TOP 30 ==="
$mcpCounts.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 30 |
    ForEach-Object { "{0,5}  {1}" -f $_.Value, $_.Key }
