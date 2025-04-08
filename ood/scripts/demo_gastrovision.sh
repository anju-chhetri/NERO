python3 ood/run_nero.py --in-dataset 'gastrovision' --num_classes 3 --model-arch 'resnet18' --weights 'path where checkpoints are saved'\
                    --seed 42 --base-dir 'path to text file to save results'\
                    --id_path_train 'path where training data is saved'\
                    --id_path_valid 'path where testing data is saved'\
                    --ood_path 'path where OOD data is saved'