# Description
Script qui permet de générer des fichiers markdown qui résument les résultats d'entrainement de modèles de classification lidar obtenus avec  
l'application [Convpoint](https://github.com/mpelchat04/ConvPoint).

# Pré-requis:   
```shell
- Matplotlib
- PyYaml
- Numpy
```  

# Contenu
Le script génère les fichiers suivants:  
```
- Dossiers des résultats brutes  
- <nom_du_projet>_config.md         # Résumé des configurations testés.  
- <nom_du_projet>_trn_metrics.md    # Résumé des métriques d'entrainement  
- <nom_du_projet>_val_metrics.md    # Résumé des métriques de validation   
- <nom_du_projet>_tst_metrics.md    # Résumé des métriques de test   
- loss.png                          # Graphiques des pertes par époques
```

# Exécution
Lancer le script:
```shell
python ./convpoint_yaml_parser.py --rootdir <dossiers_contenants_les_resultats>
```
## Structure des dossiers contenant les résultats
```
├── resultats
    └── {model_name}_{npoints}_drop{%}_{date}
        └── config.yaml
        └── metric_classwise_trn_acc.log
        └── metric_classwise_trn_fscore.log
        └── metric_classwise_trn_iou.log
        └── metric_classwise_val_acc.log
        └── metric_classwise_val_fscore.log
        └── metric_classwise_val_iou.log
        └── metric_classwise_tst_acc.log
        └── metric_classwise_tst_fscore.log
        └── metric_classwise_tst_iou.log
        └── metric_val_loss.log
        └── metric_trn_loss.log
        └── state_dict.pth
    └── {model_name}_{npoints}_drop{%}_{date}
        └── ...
```
