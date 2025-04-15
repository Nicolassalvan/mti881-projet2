import spacy
from umls.fetch_tui_from_cui import get_tuis_from_cui
import json
import xml.etree.ElementTree as ET
import os
from collections import defaultdict



# charger le modèle spacy pour une tokenisation avancée proche de celle de medmentions
nlp = spacy.load("en_core_web_sm")

def normalize_cui(cui): #juste ajouter un 0 au CUI pour que la requete API fonctionne 
    if cui[0] == "C" and len(cui) < 8 : 
        return "C" + cui[1:].zfill(7)
    return cui

def extract_cuis_from_xmi(xmi_content):
    import re
    cuis = re.findall(r'IDUMLS="([^"]+)"', xmi_content)  # extraire les CUI du .xmi basés sur l'attribut IDUMLS ou CUI (dépend de la team)
    return cuis

def extract_mentions_from_xmi(xmi_content):
    # charger le XML
    root = ET.fromstring(xmi_content)

    # Préfixes utilisés dans les balises XMI
    ns = {
        'xmi': 'http://www.omg.org/XMI',
        'cas': 'http:///uima/cas.ecore',
        'custom': 'http:///de/tudarmstadt/ukp/dkpro/core/api/semantic/type.ecore'
    }

    mentions = []

    # cherche les éléments annotés avec un attribut CUI
    for elem in root.iter():
        cui = elem.attrib.get('IDUMLS') #pour team6 -> IDUMLS, pour team3 -> CUI
        if cui:
            begin = elem.attrib.get('begin')
            end = elem.attrib.get('end')
            if begin and end:
                mentions.append({
                    "begin": int(begin),
                    "end": int(end),
                    "cui": cui
                })

    return mentions


def parse_xmi_to_medmention_format(xmi_file_path, clef_api, output_file_path, output_file_path_txt): #parser dans le format medmention
    # lire fichier XMI
    with open(xmi_file_path, "r", encoding="UTF-8") as file:
        xmi_content = file.read()
    root = ET.fromstring(xmi_content)

    # récupération du texte
    sofa_element = root.find(".//cas:Sofa", namespaces={
        "cas": "http:///uima/cas.ecore"
    })
    if sofa_element is None:
        raise ValueError("Pas de balise <cas:Sofa> dans le fichier XMI")
    text_content = sofa_element.attrib["sofaString"]

    # extraire cuis du fichier XMI
    mentions = extract_mentions_from_xmi(xmi_content)
    cuis = list({mention["cui"] for mention in mentions})

    print("les CUIs sont ici :", cuis)
    if not cuis:
        print("Aucun CUI trouvé dans le fichier XMI.")
        return

    # association CUIs avec TUIs
    cui_to_tui = {}
    for cui in cuis:
        normalized_cui = normalize_cui(cui)
        tuis = get_tuis_from_cui(normalized_cui, clef_api)
        if tuis:
            cui_to_tui[cui] = [tui for tui in tuis]
            print(f"CUIs : {cui} -> TUIs : {tuis}")
    print("cui to tui ", cui_to_tui)

    # tokenization texte avec spacy
    doc = nlp(text_content)

    # récupérer tous les tokens non-espace et leurs indices
    filtered_tokens = [(i, token) for i, token in enumerate(doc) if not token.is_space]
    tokens = [token.text for i, token in filtered_tokens]

    # initialiser les annotations BIO2
    annotations_bio2 = ["O"] * len(tokens)

    # for mention in mentions: # test
    #     print(f"Entité : {mention['cui']}, Début : {mention['begin']}, Fin : {mention['end']}")
    #     print(f"Texte correspondant : {text_content[mention['begin']:mention['end']]}")


    # print("Tokens et indices générés par SpaCy :")
    # for token in doc:
    #     print(f"Token: '{token.text}', Start: {token.idx}, End: {token.idx + len(token.text)}")
    

    for mention in mentions: #on associe chaque tui à son token 
        cui = mention["cui"]
        tuis = cui_to_tui.get(cui, [])
        if not tuis:
            continue
        tui = tuis[0] #si plusieurs tuis --> on garde uniquement le premier 

        #trouver les tokens concernés dans la liste filtrée
        mention_token_indices = []
        for idx, (orig_idx, token) in enumerate(filtered_tokens):
            token_start = token.idx
            token_end = token.idx + len(token.text)
            
            if mention["begin"] <= token_start and token_end <= mention["end"]:
                mention_token_indices.append(idx)
        
        # appliquer les annotations BIO2
        for i, token_idx in enumerate(mention_token_indices):
            if tui == "IGN":
                annotations_bio2[token_idx] = "IGN"
            else:
                if i == 0:
                    annotations_bio2[token_idx] = f"B-{tui}"
                else:
                    annotations_bio2[token_idx] = f"I-{tui}"

    # créer une structure compatible avec MedMentions
    medmention_data = {
        "tokens": tokens,
        "ner_tags": annotations_bio2
    }

    # ecrire les données dans un fichier JSON
    with open(output_file_path, "w", encoding="UTF-8") as output_file:
        json.dump(medmention_data, output_file, indent=4)

    #on écrit aussi dans un txt pour verif alignement des données
    with open(output_file_path_txt, "w", encoding="UTF-8") as output_file_txt:
        for token, annotation in zip(tokens, annotations_bio2):
            if token.strip():  # ignorer les tokens vides
                output_file_txt.write(f"{token}\t{annotation}\n")
            else:
                output_file_txt.write("\n")


    print(f"Données curées écrites dans {output_file_path} au format MedMentions")


def process_all_xmi_files(root_dir, clef_api, output_json_path):
    final_dataset = []
    
    # parcourir la structure des dossiers
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "CURATION_USER.xmi":
                xmi_path = os.path.join(root, file)
                doc_id = os.path.basename(root)  # Récupérer l'ID du document
                print(f"Traitement du fichier : {xmi_path}")
                
                # chemin temporaire pour le traitement individuel
                temp_json_path = os.path.join(root, "temp_processed.json")
                temp_txt_path = os.path.join(root, "temp_processed.txt")
                
                try:
                    # traiter le fichier XMI
                    parse_xmi_to_medmention_format(
                        xmi_path, 
                        clef_api, 
                        temp_json_path, 
                        temp_txt_path
                    )
                    
                    # Charger les données traitées
                    with open(temp_json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Séparer en phrases (en supposant que les phrases sont séparées par des points)
                        sentences = []
                        current_sentence = {"tokens": [], "ner_tags": [], "doc_id": doc_id}
                        
                        for token, tag in zip(data['tokens'], data['ner_tags']):
                            current_sentence['tokens'].append(token)
                            current_sentence['ner_tags'].append(tag)
                            
                            # Si le token est un point, on termine la phrase
                            if token == '.':
                                if current_sentence['tokens']:  # Ne pas ajouter de phrases vides
                                    sentences.append(current_sentence)
                                current_sentence = {"tokens": [], "ner_tags": [], "doc_id": doc_id}
                        
                        # Ajouter la dernière phrase si elle n'est pas vide
                        if current_sentence['tokens']:
                            sentences.append(current_sentence)
                            
                        final_dataset.extend(sentences)
                    
                    # Supprimer les fichiers temporaires
                    os.remove(temp_json_path)
                    os.remove(temp_txt_path)
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de {xmi_path}: {str(e)}")
                    continue
    
    # Sauvegarder toutes les données au format MedMentions
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4)

    # Statistiques
    print(f"\nTraitement terminé. Résultats:")
    print(f"- Nombre total de phrases: {len(final_dataset)}")
    print(f"- Nombre de documents traités: {len({item['doc_id'] for item in final_dataset})}")
    print(f"- Données sauvegardées dans {output_json_path}")


if __name__ == "__main__":
    # Configurer les chemins
    root_directory = "./etape2/data/curated_team6/" #à adapter en fonction de l'équipe à parser
    output_path = "./umls/curated_data_team6.json"
    
    # Charger la clé API
    with open("C:/Users/tecz/DDCANADA1/MTI881/mti881-projet2/umls/apikey.local", "r") as f:
        api_key = f.read().strip()
    
    # Lancer le traitement
    process_all_xmi_files(root_directory, api_key, output_path)


"""

#verif des textes un par un 
if __name__ == "__main__":
    # charger la clé API depuis un fichier local
    clef_api = open("C:/Users/tecz/DDCANADA1/MTI881/mti881-projet2/umls/apikey.local", "r", encoding="UTF-8").read()
    print("clé api :", clef_api)

    # test de fichier XMI à parser
    xmi_file_path = "C:/Users/tecz/DDCANADA1/MTI881/mti881-projet2/etape2/data/curated_team3/curation/27706165.txt/inception-document991815191211711861/CURATION_USER.xmi"
    
    #visualisation des annotations à faire 
    with open(xmi_file_path, "r", encoding="UTF-8") as file:
        xmi_content = file.read()
    root = ET.fromstring(xmi_content)

    # récupération du texte
    sofa_element = root.find(".//cas:Sofa", namespaces={
        "cas": "http:///uima/cas.ecore"
    })
    if sofa_element is None:
        raise ValueError("Pas de balise <cas:Sofa> dans le fichier XMI")
    text_content = sofa_element.attrib["sofaString"]

    # extraire cuis du fichier XMI
    mentions = extract_mentions_from_xmi(xmi_content)

    for mention in mentions: # test
        print(f"Entité : {mention['cui']}, Début : {mention['begin']}, Fin : {mention['end']}")
        print(f"Texte correspondant : {text_content[mention['begin']:mention['end']]}")

"""