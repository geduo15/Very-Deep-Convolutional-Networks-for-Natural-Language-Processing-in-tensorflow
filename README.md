# Very-Deep-Convolutional-Networks-for-Natural-Language-Processing-in-tensorflow
implement the paper" Very Deep Convolutional Networks for Natural Language Processing"(https://arxiv.org/abs/1606.01781)in tensorflow,just 9 layers(without shortcut).I change the last k-max pooling layer into max-pooling.

Parts of code is based on https://github.com/amygdala/tensorflow-workshop/tree/master/workshop_sections/cnn_text_classification ,which is based on the https://github.com/dennybritz/cnn-text-classification-tf,parts is based on https://github.com/scharmchi/char-level-cnn-tf

The data of the experiment is dbpedia, accuracy of the experiment is 0.9782,while the paper reports 0.9865
