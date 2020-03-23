# glasses_classifier

## Info
Komshanov Alexander, ML Specialization, Team: "4 ближайших соседа"
 
## Dependencies
* Python 3.6
* TensorFlow 1.15
* sklearn
* PIL
* numpy

## Data
* [Download](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) cropped Celeba images and extract them to `data/`
* Download image descriptions and save them to `data/`
* Or [download](https://drive.google.com/file/d/1K_K-ojW4MecKh1POIwMugkU3Bv10_2r3/view) preprocceced data and save it to `data/`

## Training
* Create folder `models/`
* Run `python train_validate.py`

## Inference
Run `python predict.py path_to_your_directory`
