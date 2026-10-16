import requests
import pandas as pd
import time
from pathlib import Path

# BURAYA GITHUB'DAN ALDIĞINIZ TOKENI YAPIŞTIRIN
GITHUB_TOKEN = "" 

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Görselde istenen 5 popüler repo
REPOSITORIES = [
    ("microsoft", "vscode"),
    ("facebook", "react"),
    ("tensorflow", "tensorflow"),
    ("pandas-dev", "pandas"),
    ("scikit-learn", "scikit-learn")
]

all_issues = []

for owner, repo in REPOSITORIES:
    print(f"\n{owner}/{repo} projesinden veriler çekiliyor...")
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    page = 1
    repo_count = 0
    
    # Her repodan 200 issue çekelim (Test için)
    while repo_count < 200:
        params = {"per_page": 100, "page": page, "state": "all"}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print("Hata oluştu:", response.status_code)
            break
            
        issues = response.json()
        if not issues: # Sayfalar bittiyse çık
            break
            
        for issue in issues:
            # Görseldeki kural: Pull Request'leri (PR) dışla
            if "pull_request" in issue:
                continue
                
            labels = [label["name"] for label in issue.get("labels", [])]
            
            # Görseldeki kural: İstenen verileri kaydet
            all_issues.append({
                "repo": f"{owner}/{repo}",
                "issue_number": issue["number"],
                "title": issue.get("title", ""),
                "body": issue.get("body", ""),
                "labels": ";".join(labels),
                "created_date": issue["created_at"]
            })
            
            repo_count += 1
            if repo_count >= 200:
                break
                
        page += 1
        time.sleep(1) # API limitlerine çarpmamak için kısa bir bekleme

# data/raw klasörünü oluştur ve CSV olarak kaydet
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(all_issues)
output_path = output_dir / "github_issues.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nVeri çekme tamamlandı! Toplam {len(df)} kayıt {output_path} dosyasına kaydedildi.")