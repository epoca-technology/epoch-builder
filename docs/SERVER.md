[< Back](../README.md)

# SERVER



# SSH Keys

Generate the SSH Keypair with `ssh-keygen`.

Install the SSH Public Key on the server:

`ssh-copy-id -i $HOME/.ssh/eb_id_rsa.pub epoca-worker-01@192.168.1.236`

Connect to the server:

`ssh 'epoca-worker-01@192.168.1.236'`

Fix SSH Permission issues with:

`chmod 700 ~/.ssh`

`chmod 600 ~/.ssh/*`

`ssh-add`

`ssh-add -l`


#
# Data Transfer

Transfer Files

`scp ./epoch-builder/requirements.txt epoca-worker-01@192.168.1.236:epoch-builder/requirements.txt`

Transfer Directories

`scp -r ./epoch-builder epoca-worker-01@192.168.1.236:epoch-builder`


#
# Data Removal

To remove an entire directory in the server use:

`sudo rm -r epoch-builder/`



#
# Server Information

`landscape-sysinfo`





#
# Environment Variable

Edit the .profile file:

`sudo vi ~/.profile`

Add the $PYTHONPATH Variable:

`export PYTHONPATH=/home/epoca-worker-01/epoch-builder/dist`

Refresh the file and test it:

`source ~/.profile`

`$PYTHONPATH`






#
# NVIDIA GPU

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

https://developer.nvidia.com/cuda-downloads

https://linuxconfig.org/how-to-install-cuda-on-ubuntu-20-04-focal-fossa-linux

Check for NVIDIA GPUs: 

`lspci | grep -i nvidia`

`lspci -vnnn | perl -lne 'print if /^\d+\:.+(\[\S+\:\S+\])/' | grep VGA`


## CUDA Toolkit:

Download the pin and place it in the correct directory:

`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin`

`sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600`

Grab the installer from Epoca's Drive with:

`scp ./cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb  epoca-worker-01@192.168.1.236:cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb`

Install it with:

`sudo dpkg -i cuda-repo-ubuntu2004-11-7-local_11.7.1-515.65.01-1_amd64.deb`

`sudo cp /var/cuda-repo-ubuntu2004-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/`

`wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb`

`sudo dpkg -i cuda-keyring_1.0-1_all.deb`

`sudo apt-get update`

`sudo apt-get install cuda`

`sudo apt-get install nvidia-gds`

`sudo reboot`

Verify the installation:

`sudo apt-get install cuda-drivers-470`

`nvcc --version`


## CUDNN

Grab the installer from Epoca's Drive with:

`scp ./cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb  epoca-worker-01@192.168.1.236:cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb`

Install it with:

`sudo dpkg -i cudnn-local-repo-ubuntu2004-8.5.0.96_1.0-1_amd64.deb`

`sudo cp /var/cudnn-local-repo-ubuntu2004-8.5.0.96/cudnn-local-0579404E-keyring.gpg /usr/share/keyrings/`

`sudo apt-get update`

`sudo apt-get install libcudnn8=8.5.0.96-1+cuda11.7`

`sudo apt-get install libcudnn8-dev=8.5.0.96-1+cuda11.7`

`sudo apt-get install libcudnn8-samples=8.5.0.96-1+cuda11.7`

Verify the installation:

`sudo apt-get install libfreeimage3 libfreeimage-dev`

