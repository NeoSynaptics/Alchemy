Set WShell = CreateObject("WScript.Shell")

' Check if server already running
Dim http
Set http = CreateObject("MSXML2.XMLHTTP")
On Error Resume Next
http.Open "GET", "http://localhost:8055/", False
http.Send
Dim running
running = (Err.Number = 0 And http.Status = 200)
On Error GoTo 0

' Start server minimized in background if not running
If Not running Then
    Dim serverDir
    serverDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\"))
    WShell.Run "cmd /c cd /d """ & serverDir & """ && python browser_server.py", 7, False
    WScript.Sleep 2500
End If

' Open browser — no terminal, no flash
WShell.Run "cmd /c start http://localhost:8055", 0, False
