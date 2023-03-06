


docker run -it --rm --gpus all --shm-size=192G --user $(id -u):$(id -g) --cpuset-cpus=50-74 \
-v /rsrch1/ip/msalehjahromi/codes/vesselFeatureExtraction/2_Python_extract_radiomic_features:/Code \
-v /rsrch7/wulab/Mori:/Data \
--name mori1 simplelung:Mori



…or push an existing repository from the command line
git remote add origin https://github.com/mortezasj11/vessel_features_extraction.git
git branch -M main
git push -u origin main


# if you do not commit or not type git branch -M main:
git push -u origin main
error: src refspec main does not match any.

# Password (token was giving error because when it was pasted, it was changed!!!!!!)
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

# save credentials 
git config --global credential.helper store
git pull
then after git push and inserting, it will be saved!
