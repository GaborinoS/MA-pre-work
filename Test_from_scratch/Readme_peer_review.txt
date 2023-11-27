Titel: Large-Scale Self-Supervised 
Pretraining for Railway Sound 
Detection

Author: Gabriel Schachinger 
Advisor: DI Georg Brandmayr 
FH-Prof. Priv.-Doz. Mag. Dr. David Meyer 
Date: 27.11.2023



### Short Project Description:

Grundidee: Die Problemstellung der Programmierung ist die Sound Klassifizierung von Rail Way Sound Data. Diese sind in vier Klassen zu unterteilt. Einen groben überblick kann man aus dem Notebook einsehen: analyse_labeled_data.ipynb
Wichtig ist hier zu erwähnen dass ein großer Datensatz an ungelabelten Daten und ein kleiner Datensatz an gelabeleten Railway Sound Daten. 

gelabeled: #4718
ungelabeled: cas 1.5 Terrabyxte an Daten jedoch für die ersten pretraining Methoden #44773 Datenpunkte herangezogen. (Nach bereinigung und auswahl eines Aufnahme Devicees)


Problemstellung: Aufwand für labeling von Sounddaten sehr hoch. Jedoch ist eine große anzahl an ungelabeleten Daten vorhanden die viele information beinhalten.
 Ziel ist es nun mit Hilfe von Deepleaning Ein Pretraining Model zu kreieren aus auf den ungelabelten Daten vortrainiert wird um best möglich die features aus diesen Daten zu lernen und in weiterer Folge auf den gelabelten Daten finegetuned wird.

Aufteilung in zwei Teile des Projektes:
- Supervised learning:
-- Clasification nur auf den gelabelten Daten. Mehrere Modelle, Bestes Archtektur Resnet-50

- Self-supervised Model + finetune Model:
-- 2 Phasen Model: Selbe Architektur wie aus Supervised Modell vortrainieren auf unlabeled Daten und fintuning auf labeled Daten. Um das only Supervised Model outperformen. 

Bisher 2 Pretraining Methoden angewandt:
-- Contrastive triplet Loss Model:  Mehrere Modelle, Bestes Archtektur Resnet-50

-- Masked Autoencoder Model: Mehrere Modelle, Bestes Archtektur Resnet-50

Grober Ablauf:
    1.Creation Supervised Modell. Hier sind schon sehr gute Ergebnisse erreicht worden.

    2.Pretraining Contrastive Loss: Verwendung von Pretraining Pipelines for this Model Ordenr "Pipelines"
    3.Pretraining Masked Autoencoder:  Verwendung von Pretraining Pipelines for this Model Ordenr "Pipelines"
    4.Finetuning mit beiden Modellen. Verwenden selbe finetune Pipeline. Gleich wie bei supervised Modell



#### -- Structure Notebooks:

Pipleines:
    Pipelines liegen im Python file "Pipeline_FT_SA.py"

    Dort befinden sich 4 Pipelines:

    Mypipeline: Finetunbe Pipeline: 
        Wichtige Parameter:
            - länge in Sekunden der Audio Datei (Azfrgund unterschiedliche Längen, da es für das Batching gleich lang serin muss)
            - Train ein oder aus: (auswirkungen auf Länge und masking)
            - Frequenz und Time Mask größe
        Augumentation:
            Zeit und Freq masking
        Vorgang:
            load wav file 
            resample wenn nötig
            random crop or pad: 
                Wenn kürzer als parameter "desired_length" wird der rest mit null aufgefüllt und zufällig links, mittig oder rechts abgebildet
                Wenn länger wird zufälligein teil ausgeschnitten mit der länge "desired_length"
            erstelle mel spectrogramm aus waveform
            augumentation
        Output:
            1.Melspec
