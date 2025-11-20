# Daten Push nach Git

## Variante A: HTTPS mit Personal Access Token

Arbeitsordner öffnen:

```powershell
cd F:\Dropbox\Projekte\BuBa
```

### Git Basisdaten setzen

Nur nötig, wenn diese Werte noch nicht gesetzt sind oder neu gesetzt werden sollen.

```bash
git config --global user.name "mariusbrd"
git config --global user.email "mariusbrede@gmail.com"
git config --global init.defaultBranch main
```

### Remote prüfen und auf das neue Repository setzen

```bash
git remote -v
git remote set-url origin https://github.com/mariusbrd/buba_dashboard_prod.git
```

### Anmelde Dialog sicherstellen

Windows Credential Manager verwenden:

```bash
git config --global credential.helper manager
```

### Pull und danach Push

Der Pull löst bei Bedarf den Login aus.

```bat
git pull --rebase origin main  || rem ok, wenn "up to date" oder kein Upstream
git add -A
git commit -m "update: geänderte Dateien"  || rem ok, wenn nichts zu committen
git push -u origin main
```
