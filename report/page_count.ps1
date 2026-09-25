# True page count for a .docx, by asking Word itself.
#
# Written after an estimate of "2.9 pages" turned out to be 4.3 in reality. Estimating
# pagination from word counts does not work for a document this broken up by headings,
# tables and lists, so the loop is closed properly instead: render in Word, read
# ComputeStatistics(wdStatisticPages).
#
# Usage:  powershell -ExecutionPolicy Bypass -File report/page_count.ps1 [path.docx]

param([string]$Path = "report/report.docx")

$full = (Resolve-Path $Path).Path
$word = $null
$doc  = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    # ReadOnly + AddToRecentFiles:$false so this never touches the file or the MRU list
    $doc = $word.Documents.Open($full, [ref]$false, [ref]$true)
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)   # 2 = wdStatisticPages
    $words = $doc.ComputeStatistics(0)   # 0 = wdStatisticWords
    Write-Output "PAGES=$pages"
    Write-Output "WORDS=$words"
}
catch {
    Write-Output "ERROR=$($_.Exception.Message)"
    exit 1
}
finally {
    if ($doc)  { $doc.Close([ref]$false) | Out-Null }
    if ($word) { $word.Quit() | Out-Null }
    if ($doc)  { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($word) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
}
