import os
import umls_api
import json
import pandas as pd

# Placer votre clef dans le fichier apikey.local dans ce répertoire
# Pour trouver votre clef, branchez-vous sur votre compte umls et suivez le premier lien
#   de la page https://documentation.uts.nlm.nih.gov/rest/authentication.html



def get_tuis_from_cui(id, key):
    # Lancer la requête sur l'api d'umls

    try : 
        resp = umls_api.API(api_key=key).get_cui(cui=id)

        # Sortir les TUI des relations sémantiques de l'ontologie
        if "result" in resp:
            tuis = [item["uri"].split("/")[-1] for item in resp["result"]["semanticTypes"]]
            #print(tuis)
            return tuis #!!!!!! modifié 
        else:
            #print("TUI : Aucune information trouvée")
            return ["IGN"] #!!!!! si aucun TUI n'est trouvé 
    except : 
        return ["IGN"]

# def get_tuis_from_cui(id):
#     tuis_df = pd.read_csv("tui_list.csv")
#     return tuis_df

def bio_tui_list(): #toutes les étiquettes possibles
    df = pd.read_csv("tui_list.csv")['tui']
    ret = []
    for tui in df.to_list():
        ret.append(f"B-{tui}")
        ret.append(f"I-{tui}")
    ret.append("O")
    return sorted(ret)

import requests

def get_tui_parents(tui, api_key):
    """
    Récupère les TUIs parents d'un TUI donné en interrogeant l'API REST de l'UMLS.

    :param tui: TUI pour lequel récupérer les parents.
    :param api_key: Clé API pour l'authentification à l'UMLS.
    :return: Liste des TUIs parents.
    """
    version = 'current'  # ou spécifiez une version spécifique de l'UMLS
    base_url = f'https://uts-ws.nlm.nih.gov/rest/semantic-network/{version}/TUI/{tui}'
    params = {'apiKey': api_key}

    response = requests.get(base_url, params=params)
    response_data = response.json()

    # save data 
    with open('data.json', 'w') as f:
        json.dump(response_data, f, indent=4)
    # Extraire les relations parentales
    parents = []
    for relation in response_data.get('result', {}).get('relations', []):
        if relation.get('relationLabel') == 'isa':
            parents.append(relation.get('relatedId'))

    return parents





if __name__ == "__main__":
    clef_api = open("apikey.local", "r", encoding="UTF-8").read()

    id = 'C0007107'
    get_tuis_from_cui(id, clef_api)
    print(bio_tui_list())


    # Exemple d'utilisation 
    tui_train = ['T046', 'T020', 'T019', 'T028', 'T033', 'T035', 'T038', 'T039', 'T040', 'T041', 'T042', 'T043', 'T044']
    tui = 'T047'
    # tui_parent = get_tui_parents(tui, tui_train)

    # Exemple d'utilisation
    tui = 'T046'  # Remplacez par le TUI concerné
    parents = get_tui_parents(tui, clef_api)
    print(f'Les TUIs parents de {tui} sont : {parents}')