#!/usr/bin/env python3

import argparse
import os
import time
import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1353477405624373289/-UN_D0e9qnOhK7lqVqEXLkQdCPTJn13bPNIrMn3kVRe5OfYapzS7kGN59Kn9y9mcjJcx"

def send_to_discord(image_path, message):
    """Envoie une image et un message au webhook Discord"""
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f)}
            data = {'content': message}
            response = requests.post(WEBHOOK_URL, files=files, data=data)
            response.raise_for_status()
    except Exception as e:
        print(f"Erreur lors de l'envoi de {image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Envoyer des images à un webhook Discord")
    # Modification ici pour utiliser --img_dir
    parser.add_argument('--img_dir', 
                        required=True,
                        help="Chemin du dossier contenant les images")
    parser.add_argument('--webhook_url',
                        default=WEBHOOK_URL,
                        help="URL du webhook Discord (par défaut : URL intégrée)")
    args = parser.parse_args()

    img_dir = args.img_dir.rstrip('/') + '/'
    
    # Vérifications du dossier
    if not os.path.isdir(img_dir):
        print(f"Le dossier {img_dir} n'existe pas.")
        exit(1)
        
    png_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
    if not png_files:
        print(f"Le dossier {img_dir} ne contient aucune image PNG.")
        exit(1)

    total = len(png_files)
    print(f"Fichiers trouvés : {total}")

    # Envoi des images
    for i, filename in enumerate(sorted(png_files), 1):
        full_path = os.path.join(img_dir, filename)
        message = f"Image {i}/{total} : {filename}"
        print(f"Envoi de {full_path}...")
        send_to_discord(full_path, message)
        time.sleep(1)

    print("Tous les fichiers ont été envoyés !")

if __name__ == "__main__":
    main()