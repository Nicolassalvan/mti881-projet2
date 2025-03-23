import os
import umls_api
import pandas as pd

# Placer votre clef dans le fichier apikey.local dans ce répertoire
# Pour trouver votre clef, branchez-vous sur votre compte umls et suivez le premier lien
#   de la page https://documentation.uts.nlm.nih.gov/rest/authentication.html
clef_api = open("apikey.local", "r", encoding="UTF-8").read()



def get_tuis_from_cui(id):
    # Lancer la requête sur l'api d'umls


    # Lancer la requête sur l'api d'umls

    resp = umls_api.API(api_key=clef_api).get_cui(cui=id)

    # Sortir les TUI des relations sémantiques de l'ontologie
    if "result" in resp:
        tuis = [item["uri"].split("/")[-1] for item in resp["result"]["semanticTypes"]]
        print(tuis)
    else:
        print("Aucune information trouvée")

def get_tuis_from_cui(id):
    tuis_df = pd.read_csv("tui_list.csv")
    return tuis_df

def bio_tui_list():
    df = pd.read_csv("tui_list.csv")['tui']
    ret = []
    for tui in df.to_list():
        ret.append(f"B-{tui}")
        ret.append(f"I-{tui}")
    ret.append("O")
    return sorted(ret)

if __name__ == "__main__":
    id = 'C0007107'
    get_tuis_from_cui(id)
    print(bio_tui_list())
