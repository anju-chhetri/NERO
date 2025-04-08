python3 ood/run_nero.py --in-dataset 'kvasir' --num_classes 3 --model-arch 'resnet18' --weights '/scratch/achhetri/experimentalResults/g-ood/resnet18/kvasir.pt'\
                    --seed 42 --base-dir '/users/achhetri/myWork/NERO_raw/repo_test.txt'\
                    --id_path_train '/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/kvasir/ID/train'\
                    --id_path_valid '/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/kvasir/ID/test'\
                    --ood_path '/work/FAC/HEC/DESI/yshresth/aim/achhetri/medical/kvasir/OOD'    