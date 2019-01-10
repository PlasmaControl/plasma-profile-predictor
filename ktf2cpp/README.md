===================================================
ktf2cpp - converting keras/tensorflow models to C++
===================================================


Building Tensorflow
-------------------

First create a new virtualenv (always good practice):

    conda create -n ktf2cpp python=3.6
    conda activate ktf2cpp


Install stuff you'll need:

    pip install numpy 
	pip install wheel
	sudo apt install build-essential
	sudo apt install openjdk-8-jdk

Clone tensorflow repo:

    git clone https://github.com/tensorflow/tensorflow

Install Bazel:

    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	sudo apt-get update && sudo apt-get install bazel
	sudo apt-get upgrade bazel

Make sure everything is hunky-dory (this will take a while):

    bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/lite/...

