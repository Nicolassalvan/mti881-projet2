#!/bin/bash

WEBHOOK_URL="https://discord.com/api/webhooks/1353477405624373289/-UN_D0e9qnOhK7lqVqEXLkQdCPTJn13bPNIrMn3kVRe5OfYapzS7kGN59Kn9y9mcjJcx"
# MESSAGE="Résultats de l'entrainement"


# Vérifie si un argument (chemin du dossier) a été fourni
if [ -z "$1" ]; then
    echo "Usage : $0 <chemin_du_dossier_images>"
    exit 1
fi
IMG_DIR="$1"
# Vérifie si le dossier img/ existe
if [ ! -d "$IMG_DIR" ]; then
    echo "Le dossier $IMG_DIR n'existe pas."
    exit 1
fi
# Vérifie si le dossier img/ est vide
if [ -z "$(ls -A $IMG_DIR)" ]; then
    echo "Le dossier $IMG_DIR est vide. Veuillez y placer des images."
    exit 1
fi
# Rajouter un / à la fin du chemin si ce n'est pas déjà fait
IMG_DIR="${IMG_DIR%/}/"

files=("$IMG_DIR"*.png)
echo "Fichiers trouvés : ${#files[@]}"
total_images=${#files[@]}

# Boucle sur tous les fichiers image du dossier spécifié
for (( i=0; i<total_images; i++ )); do
    file="${files[$i]}"
    MESSAGE="Image $(($i + 1))/$total_images : $file"
    echo "Envoi de l'image $file..."
    # echo "Message : $MESSAGE"
    if [ -f "$file" ]; then
        echo "Envoi de $file..."
        curl -X POST "$WEBHOOK_URL" \
             -F "content=$MESSAGE" \
             -F "file=@$file"
        sleep 1  # Pause pour éviter d'éventuels ratés
    fi
done

echo "Tous les fichiers ont été envoyés !"
exit 0
