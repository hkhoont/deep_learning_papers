# deep_learning_papers
Deep Learning - Papers and articles
## 1.0 Book

Bengio, Yoshua, Ian J. Goodfellow, and Aaron Courville. "**Deep learning**." An MIT Press book. (2015). [[html]](http://www.deeplearningbook.org/) **(Deep Learning Bible, you can read this book while reading following papers.)** 

## 1.1 Survey

LeCun, Yann, Yoshua Bengio, and Geoffrey Hinton. "**Deep learning**." Nature 521.7553 (2015): 436-444. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) **(Three Giants' Survey)** 

## 1.2 Deep Belief Network(DBN)(Milestone of Deep Learning Eve)

Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "**A fast learning algorithm for deep belief nets**." Neural computation 18.7 (2006): 1527-1554. [[pdf]](http://www.cs.toronto.edu/~hinton/absps/ncfast.pdf)**(Deep Learning Eve)** 

Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "**Reducing the dimensionality of data with neural networks**." Science 313.5786 (2006): 504-507. [[pdf]](http://www.cs.toronto.edu/~hinton/science.pdf) **(Milestone, Show the promise of deep learning)**

## 1.3 ImageNet Evolution（Deep Learning broke out from here）

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "**Imagenet classification with deep convolutional neural networks**." Advances in neural information processing systems. 2012. [[pdf]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) **(AlexNet, Deep Learning Breakthrough)** 

Simonyan, Karen, and Andrew Zisserman. "**Very deep convolutional networks for large-scale image recognition**." arXiv preprint arXiv:1409.1556 (2014). [[pdf]](https://arxiv.org/pdf/1409.1556.pdf) **(VGGNet,Neural Networks become very deep!)** 

Szegedy, Christian, et al. "**Going deeper with convolutions**." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf) **(GoogLeNet)** 

He, Kaiming, et al. "**Deep residual learning for image recognition**." arXiv preprint arXiv:1512.03385 (2015). [[pdf]](https://arxiv.org/pdf/1512.03385.pdf) **(ResNet,Very very deep networks, CVPR best paper)** 

## 1.4 Speech Recognition Evolution

Hinton, Geoffrey, et al. "**Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups**." IEEE Signal Processing Magazine 29.6 (2012): 82-97. [[pdf]](http://cs224d.stanford.edu/papers/maas_paper.pdf) **(Breakthrough in speech recognition)**

Graves, Alex, Abdel-rahman Mohamed, and Geoffrey Hinton. "**Speech recognition with deep recurrent neural networks**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1303.5778.pdf) **(RNN)**

Graves, Alex, and Navdeep Jaitly. "**Towards End-To-End Speech Recognition with Recurrent Neural Networks**." ICML. Vol. 14. 2014. [[pdf]](http://www.jmlr.org/proceedings/papers/v32/graves14.pdf)

Sak, Haşim, et al. "**Fast and accurate recurrent neural network acoustic models for speech recognition**." arXiv preprint arXiv:1507.06947 (2015). [[pdf]](http://arxiv.org/pdf/1507.06947) **(Google Speech Recognition System)** 

Amodei, Dario, et al. "**Deep speech 2: End-to-end speech recognition in english and mandarin**." arXiv preprint arXiv:1512.02595 (2015). [[pdf]](https://arxiv.org/pdf/1512.02595.pdf) **(Baidu Speech Recognition System)** 

W. Xiong, J. Droppo, X. Huang, F. Seide, M. Seltzer, A. Stolcke, D. Yu, G. Zweig "**Achieving Human Parity in Conversational Speech Recognition**." arXiv preprint arXiv:1610.05256 (2016). [[pdf]](https://arxiv.org/pdf/1610.05256v1) **(State-of-the-art in speech recognition, Microsoft)**


#2 Deep Learning Method

## 2.1 Model

Hinton, Geoffrey E., et al. "**Improving neural networks by preventing co-adaptation of feature detectors**." arXiv preprint arXiv:1207.0580 (2012). [[pdf]](https://arxiv.org/pdf/1207.0580.pdf) **(Dropout)** 

Srivastava, Nitish, et al. "**Dropout: a simple way to prevent neural networks from overfitting**." Journal of Machine Learning Research 15.1 (2014): 1929-1958. [[pdf]](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) 

Ioffe, Sergey, and Christian Szegedy. "**Batch normalization: Accelerating deep network training by reducing internal covariate shift**." arXiv preprint arXiv:1502.03167 (2015). [[pdf]](http://arxiv.org/pdf/1502.03167) **(An outstanding Work in 2015)**

Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "**Layer normalization**." arXiv preprint arXiv:1607.06450 (2016). [[pdf]](https://arxiv.org/pdf/1607.06450.pdf?utm_source=sciontist.com&utm_medium=refer&utm_campaign=promote) **(Update of Batch Normalization)**

Courbariaux, Matthieu, et al. "**Binarized Neural Networks: Training Neural Networks with Weights and Activations Constrained to+ 1 or−1**." [[pdf]](https://pdfs.semanticscholar.org/f832/b16cb367802609d91d400085eb87d630212a.pdf) **(New Model,Fast)**

Jaderberg, Max, et al. "**Decoupled neural interfaces using synthetic gradients**." arXiv preprint arXiv:1608.05343 (2016). [[pdf]](https://arxiv.org/pdf/1608.05343) **(Innovation of Training Method,Amazing Work)** 

Chen, Tianqi, Ian Goodfellow, and Jonathon Shlens. "Net2net: Accelerating learning via knowledge transfer." arXiv preprint arXiv:1511.05641 (2015). [[pdf]](https://arxiv.org/abs/1511.05641) **(Modify previously trained network to reduce training epochs)**

Wei, Tao, et al. "Network Morphism." arXiv preprint arXiv:1603.01670 (2016). [[pdf]](https://arxiv.org/abs/1603.01670) **(Modify previously trained network to reduce training epochs)**

## 2.2 Optimization

Sutskever, Ilya, et al. "**On the importance of initialization and momentum in deep learning**." ICML (3) 28 (2013): 1139-1147. [[pdf]](http://www.jmlr.org/proceedings/papers/v28/sutskever13.pdf) **(Momentum optimizer)** 

Kingma, Diederik, and Jimmy Ba. "**Adam: A method for stochastic optimization**." arXiv preprint arXiv:1412.6980 (2014). [[pdf]](http://arxiv.org/pdf/1412.6980) **(Maybe used most often currently)**

Andrychowicz, Marcin, et al. "**Learning to learn by gradient descent by gradient descent**." arXiv preprint arXiv:1606.04474 (2016). [[pdf]](https://arxiv.org/pdf/1606.04474) **(Neural Optimizer,Amazing Work)** 

Han, Song, Huizi Mao, and William J. Dally. "**Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding**." CoRR, abs/1510.00149 2 (2015). [[pdf]](https://pdfs.semanticscholar.org/5b6c/9dda1d88095fa4aac1507348e498a1f2e863.pdf) **(ICLR best paper, new direction to make NN running fast,DeePhi Tech Startup)**

Iandola, Forrest N., et al. "**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size**." arXiv preprint arXiv:1602.07360 (2016). [[pdf]](http://arxiv.org/pdf/1602.07360) **(Also a new direction to optimize NN,DeePhi Tech Startup)**

## 2.3 Unsupervised Learning / Deep Generative Model

Le, Quoc V. "**Building high-level features using large scale unsupervised learning**." 2013 IEEE international conference on acoustics, speech and signal processing. IEEE, 2013. [[pdf]](http://arxiv.org/pdf/1112.6209.pdf&embed) **(Milestone, Andrew Ng, Google Brain Project, Cat)**


Kingma, Diederik P., and Max Welling. "**Auto-encoding variational bayes**." arXiv preprint arXiv:1312.6114 (2013). [[pdf]](http://arxiv.org/pdf/1312.6114) **(VAE)**

Goodfellow, Ian, et al. "**Generative adversarial nets**." Advances in Neural Information Processing Systems. 2014. [[pdf]](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) **(GAN,super cool idea)**

Radford, Alec, Luke Metz, and Soumith Chintala. "**Unsupervised representation learning with deep convolutional generative adversarial networks**." arXiv preprint arXiv:1511.06434 (2015). [[pdf]](http://arxiv.org/pdf/1511.06434) **(DCGAN)**

Gregor, Karol, et al. "**DRAW: A recurrent neural network for image generation**." arXiv preprint arXiv:1502.04623 (2015). [[pdf]](http://jmlr.org/proceedings/papers/v37/gregor15.pdf) **(VAE with attention, outstanding work)** 

Oord, Aaron van den, Nal Kalchbrenner, and Koray Kavukcuoglu. "**Pixel recurrent neural networks**." arXiv preprint arXiv:1601.06759 (2016). [[pdf]](http://arxiv.org/pdf/1601.06759) **(PixelRNN)**

Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders." arXiv preprint arXiv:1606.05328 (2016). [[pdf]](https://arxiv.org/pdf/1606.05328) **(PixelCNN)**

## 2.4 RNN / Sequence-to-Sequence Model

Graves, Alex. "**Generating sequences with recurrent neural networks**." arXiv preprint arXiv:1308.0850 (2013). [[pdf]](http://arxiv.org/pdf/1308.0850) **(LSTM, very nice generating result, show the power of RNN)** 

Cho, Kyunghyun, et al. "**Learning phrase representations using RNN encoder-decoder for statistical machine translation**." arXiv preprint arXiv:1406.1078 (2014). [[pdf]](http://arxiv.org/pdf/1406.1078) **(First Seq-to-Seq Paper)** 

Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "**Sequence to sequence learning with neural networks**." Advances in neural information processing systems. 2014. [[pdf]](https://arxiv.org/pdf/1409.3215.pdf) **(Outstanding Work)** 

Bahdanau, Dzmitry, KyungHyun Cho, and Yoshua Bengio. "**Neural Machine Translation by Jointly Learning to Align and Translate**." arXiv preprint arXiv:1409.0473 (2014). [[pdf]](https://arxiv.org/pdf/1409.0473v7.pdf)

Vinyals, Oriol, and Quoc Le. "**A neural conversational model**." arXiv preprint arXiv:1506.05869 (2015). [[pdf]](http://arxiv.org/pdf/1506.05869.pdf%20(http://arxiv.org/pdf/1506.05869.pdf)) **(Seq-to-Seq on Chatbot)**

## 2.5 Neural Turing Machine

Graves, Alex, Greg Wayne, and Ivo Danihelka. "**Neural turing machines**." arXiv preprint arXiv:1410.5401 (2014). [[pdf]](http://arxiv.org/pdf/1410.5401.pdf) **(Basic Prototype of Future Computer)** 

Zaremba, Wojciech, and Ilya Sutskever. "**Reinforcement learning neural Turing machines**." arXiv preprint arXiv:1505.00521 362 (2015). [[pdf]](https://pdfs.semanticscholar.org/f10e/071292d593fef939e6ef4a59baf0bb3a6c2b.pdf) 

Weston, Jason, Sumit Chopra, and Antoine Bordes. "**Memory networks**." arXiv preprint arXiv:1410.3916 (2014). [[pdf]](http://arxiv.org/pdf/1410.3916)

Sukhbaatar, Sainbayar, Jason Weston, and Rob Fergus. "**End-to-end memory networks**." Advances in neural information processing systems. 2015. [[pdf]](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf) 

Vinyals, Oriol, Meire Fortunato, and Navdeep Jaitly. "**Pointer networks**." Advances in Neural Information Processing Systems. 2015. [[pdf]](http://papers.nips.cc/paper/5866-pointer-networks.pdf)

Graves, Alex, et al. "**Hybrid computing using a neural network with dynamic external memory**." Nature (2016). [[pdf]](https://www.dropbox.com/s/0a40xi702grx3dq/2016-graves.pdf) **(Milestone,combine above papers' ideas)** 

## 2.6 Deep Reinforcement Learning

Mnih, Volodymyr, et al. "**Playing atari with deep reinforcement learning**." arXiv preprint arXiv:1312.5602 (2013). [[pdf]](http://arxiv.org/pdf/1312.5602.pdf)) **(First Paper named deep reinforcement learning)** 

Mnih, Volodymyr, et al. "**Human-level control through deep reinforcement learning**." Nature 518.7540 (2015): 529-533. [[pdf]](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) **(Milestone)** 

Wang, Ziyu, Nando de Freitas, and Marc Lanctot. "**Dueling network architectures for deep reinforcement learning**." arXiv preprint arXiv:1511.06581 (2015). [[pdf]](http://arxiv.org/pdf/1511.06581) **(ICLR best paper,great idea)** 

Mnih, Volodymyr, et al. "**Asynchronous methods for deep reinforcement learning**." arXiv preprint arXiv:1602.01783 (2016). [[pdf]](http://arxiv.org/pdf/1602.01783) **(State-of-the-art method)** 

Lillicrap, Timothy P., et al. "**Continuous control with deep reinforcement learning**." arXiv preprint arXiv:1509.02971 (2015). [[pdf]](http://arxiv.org/pdf/1509.02971) **(DDPG)** 

Gu, Shixiang, et al. "**Continuous Deep Q-Learning with Model-based Acceleration**." arXiv preprint arXiv:1603.00748 (2016). [[pdf]](http://arxiv.org/pdf/1603.00748) **(NAF)**

Schulman, John, et al. "**Trust region policy optimization**." CoRR, abs/1502.05477 (2015). [[pdf]](http://www.jmlr.org/proceedings/papers/v37/schulman15.pdf) **(TRPO)**

Silver, David, et al. "**Mastering the game of Go with deep neural networks and tree search**." Nature 529.7587 (2016): 484-489. [[pdf]](http://willamette.edu/~levenick/cs448/goNature.pdf) **(AlphaGo)**

## 2.7 Deep Transfer Learning / Lifelong Learning / especially for RL

Bengio, Yoshua. "**Deep Learning of Representations for Unsupervised and Transfer Learning**." ICML Unsupervised and Transfer Learning 27 (2012): 17-36. [[pdf]](http://www.jmlr.org/proceedings/papers/v27/bengio12a/bengio12a.pdf) **(A Tutorial)**

Silver, Daniel L., Qiang Yang, and Lianghao Li. "**Lifelong Machine Learning Systems: Beyond Learning Algorithms**." AAAI Spring Symposium: Lifelong Machine Learning. 2013. [[pdf]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.696.7800&rep=rep1&type=pdf) **(A brief discussion about lifelong learning)**  

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "**Distilling the knowledge in a neural network**." arXiv preprint arXiv:1503.02531 (2015). [[pdf]](http://arxiv.org/pdf/1503.02531) **(Godfather's Work)** 

Rusu, Andrei A., et al. "**Policy distillation**." arXiv preprint arXiv:1511.06295 (2015). [[pdf]](http://arxiv.org/pdf/1511.06295) **(RL domain)**

Parisotto, Emilio, Jimmy Lei Ba, and Ruslan Salakhutdinov. "**Actor-mimic: Deep multitask and transfer reinforcement learning**." arXiv preprint arXiv:1511.06342 (2015). [[pdf]](http://arxiv.org/pdf/1511.06342) **(RL domain)** 

Rusu, Andrei A., et al. "**Progressive neural networks**." arXiv preprint arXiv:1606.04671 (2016). [[pdf]](https://arxiv.org/pdf/1606.04671) **(Outstanding Work, A novel idea)**


## 2.8 One Shot Deep Learning

Lake, Brenden M., Ruslan Salakhutdinov, and Joshua B. Tenenbaum. "**Human-level concept learning through probabilistic program induction**." Science 350.6266 (2015): 1332-1338. [[pdf]](http://clm.utexas.edu/compjclub/wp-content/uploads/2016/02/lake2015.pdf) **(No Deep Learning,but worth reading)**

Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "**Siamese Neural Networks for One-shot Image Recognition**."(2015) [[pdf]](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf)

Santoro, Adam, et al. "**One-shot Learning with Memory-Augmented Neural Networks**." arXiv preprint arXiv:1605.06065 (2016). [[pdf]](http://arxiv.org/pdf/1605.06065) **(A basic step to one shot learning)**

Vinyals, Oriol, et al. "**Matching Networks for One Shot Learning**." arXiv preprint arXiv:1606.04080 (2016). [[pdf]](https://arxiv.org/pdf/1606.04080)

Hariharan, Bharath, and Ross Girshick. "**Low-shot visual object recognition**." arXiv preprint arXiv:1606.02819 (2016). [[pdf]](http://arxiv.org/pdf/1606.02819) **(A step to large data)**


# 3 Applications

## 3.1 NLP(Natural Language Processing)

Antoine Bordes, et al. "**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**." AISTATS(2012) [[pdf]](https://www.hds.utc.fr/~bordesan/dokuwiki/lib/exe/fetch.php?id=en%3Apubli&cache=cache&media=en:bordes12aistats.pdf)

Mikolov, et al. "**Distributed representations of words and phrases and their compositionality**." ANIPS(2013): 3111-3119 [[pdf]](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) **(word2vec)**

Sutskever, et al. "**“Sequence to sequence learning with neural networks**." ANIPS(2014) [[pdf]](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Ankit Kumar, et al. "**“Ask Me Anything: Dynamic Memory Networks for Natural Language Processing**." arXiv preprint arXiv:1506.07285(2015) [[pdf]](https://arxiv.org/abs/1506.07285)

Yoon Kim, et al. "**Character-Aware Neural Language Models**." NIPS(2015) arXiv preprint arXiv:1508.06615(2015) [[pdf]](https://arxiv.org/abs/1508.06615)

Jason Weston, et al. "**Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks**." arXiv preprint arXiv:1502.05698(2015) [[pdf]](https://arxiv.org/abs/1502.05698) **(bAbI tasks)**

Karl Moritz Hermann, et al. "**Teaching Machines to Read and Comprehend**." arXiv preprint arXiv:1506.03340(2015) [[pdf]](https://arxiv.org/abs/1506.03340) **(CNN/DailyMail cloze style questions)**

Alexis Conneau, et al. "**Very Deep Convolutional Networks for Natural Language Processing**." arXiv preprint arXiv:1606.01781(2016) [[pdf]](https://arxiv.org/abs/1606.01781) **(state-of-the-art in text classification)** 

Armand Joulin, et al. "**Bag of Tricks for Efficient Text Classification**." arXiv preprint arXiv:1607.01759(2016) [[pdf]](https://arxiv.org/abs/1607.01759) **(slightly worse than state-of-the-art, but a lot faster)** 

### 3.1.1 Machine Translation & Transliteration, Sequence-to-Sequence Models

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le Sequence to Sequence Learning with Neural Networks, NIPS 2014 [[pdf]](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

Oriol Vinyals, Quoc Le: A Neural Conversation Model, 2015 [[pdf]](https://arxiv.org/pdf/1506.05869.pdf)

Dzmitry Bahdanau, et al.: Neural Machine Translation by Jointly Learning to Align and Translate, 2014 [[pdf]](https://arxiv.org/pdf/1409.0473.pdf)

Minh-Thang Luong, et al.: Effective Approaches to Attention-based Neural Machine Translation, 2015 [[pdf]](https://arxiv.org/pdf/1508.04025.pdf)

Yonghui Wu, et al.: Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, 2016 [[pdf]](https://arxiv.org/pdf/1609.08144.pdf)

Jonas Gehring, et al.: Convolutional Sequence to Sequence Learning, 2017 [[pdf]](https://arxiv.org/pdf/1705.03122.pdf)

Ashish Vaswani, et al.: Attention Is All You Need, 2017 [[pdf]](https://arxiv.org/pdf/1706.03762.pdf)

## 3.2 Object Detection

Szegedy, Christian, Alexander Toshev, and Dumitru Erhan. "**Deep neural networks for object detection**." Advances in Neural Information Processing Systems. 2013. [[pdf]](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

Girshick, Ross, et al. "**Rich feature hierarchies for accurate object detection and semantic segmentation**." Proceedings of the IEEE conference on computer vision and pattern recognition. 2014. [[pdf]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) **(RCNN)** 

He, Kaiming, et al. "**Spatial pyramid pooling in deep convolutional networks for visual recognition**." European Conference on Computer Vision. Springer International Publishing, 2014. [[pdf]](http://arxiv.org/pdf/1406.4729) **(SPPNet)** 

Girshick, Ross. "**Fast r-cnn**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](https://pdfs.semanticscholar.org/8f67/64a59f0d17081f2a2a9d06f4ed1cdea1a0ad.pdf) 

Ren, Shaoqing, et al. "**Faster R-CNN: Towards real-time object detection with region proposal networks**." Advances in neural information processing systems. 2015. [[pdf]](https://arxiv.org/pdf/1506.01497.pdf)

Redmon, Joseph, et al. "**You only look once: Unified, real-time object detection**." arXiv preprint arXiv:1506.02640 (2015). [[pdf]](http://homes.cs.washington.edu/~ali/papers/YOLO.pdf) **(YOLO,Oustanding Work, really practical)**

Liu, Wei, et al. "**SSD: Single Shot MultiBox Detector**." arXiv preprint arXiv:1512.02325 (2015). [[pdf]](http://arxiv.org/pdf/1512.02325)

Dai, Jifeng, et al. "**R-FCN: Object Detection via
Region-based Fully Convolutional Networks**." arXiv preprint arXiv:1605.06409 (2016). [[pdf]](https://arxiv.org/abs/1605.06409)

He, Gkioxari, et al. "**Mask R-CNN**" arXiv preprint arXiv:1703.06870 (2017). [[pdf]](https://arxiv.org/abs/1703.06870)

## 3.3 Visual Tracking

Wang, Naiyan, and Dit-Yan Yeung. "**Learning a deep compact image representation for visual tracking**." Advances in neural information processing systems. 2013. [[pdf]](http://papers.nips.cc/paper/5192-learning-a-deep-compact-image-representation-for-visual-tracking.pdf) **(First Paper to do visual tracking using Deep Learning,DLT Tracker)** 

Wang, Naiyan, et al. "**Transferring rich feature hierarchies for robust visual tracking**." arXiv preprint arXiv:1501.04587 (2015). [[pdf]](http://arxiv.org/pdf/1501.04587) **(SO-DLT)**

Wang, Lijun, et al. "**Visual tracking with fully convolutional networks**." Proceedings of the IEEE International Conference on Computer Vision. 2015. [[pdf]](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Wang_Visual_Tracking_With_ICCV_2015_paper.pdf) **(FCNT)** 

Held, David, Sebastian Thrun, and Silvio Savarese. "**Learning to Track at 100 FPS with Deep Regression Networks**." arXiv preprint arXiv:1604.01802 (2016). [[pdf]](http://arxiv.org/pdf/1604.01802) **(GOTURN,Really fast as a deep learning method,but still far behind un-deep-learning methods)**

Bertinetto, Luca, et al. "**Fully-Convolutional Siamese Networks for Object Tracking**." arXiv preprint arXiv:1606.09549 (2016). [[pdf]](https://arxiv.org/pdf/1606.09549) **(SiameseFC,New state-of-the-art for real-time object tracking)** 

Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. "**Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking**." ECCV (2016) [[pdf]](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf) **(C-COT)** 

Nam, Hyeonseob, Mooyeol Baek, and Bohyung Han. "**Modeling and Propagating CNNs in a Tree Structure for Visual Tracking**." arXiv preprint arXiv:1608.07242 (2016). [[pdf]](https://arxiv.org/pdf/1608.07242) **(VOT2016 Winner,TCNN)** 

## 3.4 Image Caption
Farhadi,Ali,etal. "**Every picture tells a story: Generating sentences from images**". In Computer VisionECCV 2010. Springer Berlin Heidelberg:15-29, 2010. [[pdf]](https://www.cs.cmu.edu/~afarhadi/papers/sentence.pdf) 

Kulkarni, Girish, et al. "**Baby talk: Understanding and generating image descriptions**". In Proceedings of the 24th CVPR, 2011. [[pdf]](http://tamaraberg.com/papers/generation_cvpr11.pdf)

Vinyals, Oriol, et al. "**Show and tell: A neural image caption generator**". In arXiv preprint arXiv:1411.4555, 2014. [[pdf]](https://arxiv.org/pdf/1411.4555.pdf)

Donahue, Jeff, et al. "**Long-term recurrent convolutional networks for visual recognition and description**". In arXiv preprint arXiv:1411.4389 ,2014. [[pdf]](https://arxiv.org/pdf/1411.4389.pdf)

Karpathy, Andrej, and Li Fei-Fei. "**Deep visual-semantic alignments for generating image descriptions**". In arXiv preprint arXiv:1412.2306, 2014. [[pdf]](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)

Karpathy, Andrej, Armand Joulin, and Fei Fei F. Li. "**Deep fragment embeddings for bidirectional image sentence mapping**". In Advances in neural information processing systems, 2014. [[pdf]](https://arxiv.org/pdf/1406.5679v1.pdf)

Fang, Hao, et al. "**From captions to visual concepts and back**". In arXiv preprint arXiv:1411.4952, 2014. [[pdf]](https://arxiv.org/pdf/1411.4952v3.pdf)

Chen, Xinlei, and C. Lawrence Zitnick. "**Learning a recurrent visual representation for image caption generation**". In arXiv preprint arXiv:1411.5654, 2014. [[pdf]](https://arxiv.org/pdf/1411.5654v1.pdf)

Mao, Junhua, et al. "**Deep captioning with multimodal recurrent neural networks (m-rnn)**". In arXiv preprint arXiv:1412.6632, 2014. [[pdf]](https://arxiv.org/pdf/1412.6632v5.pdf)

Xu, Kelvin, et al. "**Show, attend and tell: Neural image caption generation with visual attention**". In arXiv preprint arXiv:1502.03044, 2015. [[pdf]](https://arxiv.org/pdf/1502.03044v3.pdf)

## 3.5 Machine Translation

> Some milestone papers are listed in RNN / Seq-to-Seq topic.

Luong, Minh-Thang, et al. "**Addressing the rare word problem in neural machine translation**." arXiv preprint arXiv:1410.8206 (2014). [[pdf]](http://arxiv.org/pdf/1410.8206)

Sennrich, et al. "**Neural Machine Translation of Rare Words with Subword Units**". In arXiv preprint arXiv:1508.07909, 2015. [[pdf]](https://arxiv.org/pdf/1508.07909.pdf)

Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "**Effective approaches to attention-based neural machine translation**." arXiv preprint arXiv:1508.04025 (2015). [[pdf]](http://arxiv.org/pdf/1508.04025)

Chung, et al. "**A Character-Level Decoder without Explicit Segmentation for Neural Machine Translation**". In arXiv preprint arXiv:1603.06147, 2016. [[pdf]](https://arxiv.org/pdf/1603.06147.pdf)

Lee, et al. "**Fully Character-Level Neural Machine Translation without Explicit Segmentation**". In arXiv preprint arXiv:1610.03017, 2016. [[pdf]](https://arxiv.org/pdf/1610.03017.pdf)

Wu, Schuster, Chen, Le, et al. "**Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation**". In arXiv preprint arXiv:1609.08144v2, 2016. [[pdf]](https://arxiv.org/pdf/1609.08144v2.pdf) **(Milestone)** 

## 3.6 Robotics

Koutník, Jan, et al. "**Evolving large-scale neural networks for vision-based reinforcement learning**." Proceedings of the 15th annual conference on Genetic and evolutionary computation. ACM, 2013. [[pdf]](http://repository.supsi.ch/4550/1/koutnik2013gecco.pdf)

Levine, Sergey, et al. "**End-to-end training of deep visuomotor policies**." Journal of Machine Learning Research 17.39 (2016): 1-40. [[pdf]](http://www.jmlr.org/papers/volume17/15-522/15-522.pdf)

Pinto, Lerrel, and Abhinav Gupta. "**Supersizing self-supervision: Learning to grasp from 50k tries and 700 robot hours**." arXiv preprint arXiv:1509.06825 (2015). [[pdf]](http://arxiv.org/pdf/1509.06825)

Levine, Sergey, et al. "**Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection**." arXiv preprint arXiv:1603.02199 (2016). [[pdf]](http://arxiv.org/pdf/1603.02199) 

Zhu, Yuke, et al. "**Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning**." arXiv preprint arXiv:1609.05143 (2016). [[pdf]](https://arxiv.org/pdf/1609.05143) 

Yahya, Ali, et al. "**Collective Robot Reinforcement Learning with Distributed Asynchronous Guided Policy Search**." arXiv preprint arXiv:1610.00673 (2016). [[pdf]](https://arxiv.org/pdf/1610.00673)

Gu, Shixiang, et al. "**Deep Reinforcement Learning for Robotic Manipulation**." arXiv preprint arXiv:1610.00633 (2016). [[pdf]](https://arxiv.org/pdf/1610.00633)

A Rusu, M Vecerik, Thomas Rothörl, N Heess, R Pascanu, R Hadsell."**Sim-to-Real Robot Learning from Pixels with Progressive Nets**." arXiv preprint arXiv:1610.04286 (2016). [[pdf]](https://arxiv.org/pdf/1610.04286.pdf) 

Mirowski, Piotr, et al. "**Learning to navigate in complex environments**." arXiv preprint arXiv:1611.03673 (2016). [[pdf]](https://arxiv.org/pdf/1611.03673)

## 3.7 Art

Mordvintsev, Alexander; Olah, Christopher; Tyka, Mike (2015). "**Inceptionism: Going Deeper into Neural Networks**". Google Research. [[html]](https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) **(Deep Dream)**


Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "**A neural algorithm of artistic style**." arXiv preprint arXiv:1508.06576 (2015). [[pdf]](http://arxiv.org/pdf/1508.06576) **(Outstanding Work, most successful method currently)** 

Zhu, Jun-Yan, et al. "**Generative Visual Manipulation on the Natural Image Manifold**." European Conference on Computer Vision. Springer International Publishing, 2016. [[pdf]](https://arxiv.org/pdf/1609.03552) **(iGAN)** 

Champandard, Alex J. "**Semantic Style Transfer and Turning Two-Bit Doodles into Fine Artworks**." arXiv preprint arXiv:1603.01768 (2016). [[pdf]](http://arxiv.org/pdf/1603.01768) **(Neural Doodle)** 

Zhang, Richard, Phillip Isola, and Alexei A. Efros. "**Colorful Image Colorization**." arXiv preprint arXiv:1603.08511 (2016). [[pdf]](http://arxiv.org/pdf/1603.08511)

Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "**Perceptual losses for real-time style transfer and super-resolution**." arXiv preprint arXiv:1603.08155 (2016). [[pdf]](https://arxiv.org/pdf/1603.08155.pdf)

Vincent Dumoulin, Jonathon Shlens and Manjunath Kudlur. "**A learned representation for artistic style**." arXiv preprint arXiv:1610.07629 (2016). [[pdf]](https://arxiv.org/pdf/1610.07629v1.pdf)

Gatys, Leon and Ecker, et al."**Controlling Perceptual Factors in Neural Style Transfer**." arXiv preprint arXiv:1611.07865 (2016). [[pdf]](https://arxiv.org/pdf/1611.07865.pdf) **(control style transfer over spatial location,colour information and across spatial scale)**

Ulyanov, Dmitry and Lebedev, Vadim, et al. "**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**." arXiv preprint arXiv:1603.03417(2016). [[pdf]](http://arxiv.org/abs/1603.03417) **(texture generation and style transfer)**


## 3.8 Object Segmentation

J. Long, E. Shelhamer, and T. Darrell, “**Fully convolutional networks for semantic segmentation**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1411.4038v2.pdf)

L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. "**Semantic image segmentation with deep convolutional nets and fully connected crfs**." In ICLR, 2015. [[pdf]](https://arxiv.org/pdf/1606.00915v1.pdf) 

Pinheiro, P.O., Collobert, R., Dollar, P. "**Learning to segment object candidates.**" In: NIPS. 2015. [[pdf]](https://arxiv.org/pdf/1506.06204v2.pdf)

Dai, J., He, K., Sun, J. "**Instance-aware semantic segmentation via multi-task network cascades**." in CVPR. 2016 [[pdf]](https://arxiv.org/pdf/1512.04412v1.pdf)

Dai, J., He, K., Sun, J. "**Instance-sensitive Fully Convolutional Networks**." arXiv preprint arXiv:1603.08678 (2016). [[pdf]](https://arxiv.org/pdf/1603.08678v1.pdf)

# 4 Papers helpful to understand mathematics and logic in Deep Learning book - Ian goodfellow

### 4.1 CNN 
**[1]** Vincent Dumoulin, Francesco Visin, “**A guide to convolution arithmetic for deep learning**.” in CVPR, 2015. [[pdf]](https://arxiv.org/pdf/1603.07285.pdf)

### 4.2 RNN

“**BackPropagation Through Time**.” [[pdf]](http://ir.hit.edu.cn/~jguo/docs/notes/bptt.pdf) 

“**Multi-Dimensional Recurrent Neural Networks**” Useful when you want to understand what happens to image when you apply Multidirectional RNN. [[pdf]](https://arxiv.org/pdf/0705.2011.pdf)

“**Training RNN**” Ilya Sutskever: A thesis submitted in conformity with the requirements for the degree of Doctor of Philosophy Graduate Department of Computer Science University of Toronto
 [[pdf]](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)
 
“**Learning long term dependency with gradient descent in RNN is difficult**”
 [[pdf]]( http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf)
 
“**Deriving Back Propagation on simple RNN/LSTM**”
 [[Link]](https://towardsdatascience.com/back-to-basics-deriving-back-propagation-on-simple-rnn-lstm-feat-aidan-gomez-c7f286ba973d)
 
Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Yoshua Bengio “**How to Construct Deep Recurrent Neural Networks**”
[[pdf]](https://arxiv.org/pdf/1312.6026.pdf)

### 4.3 LSTM

“**LSTM**.” This paper gives a mathematical  understanding of vanishing gradient problem in RNN and subsequent solution by developing LSTM [[pdf]](https://www.bioinf.jku.at/publications/older/2604.pdf)

Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, Adam Kalai “**Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings**.” Removing Bias in word embeddings [[pdf]](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)

### 4.4 RNN/LSTM Applications

Elman Mansimov, Emilio Parisotto, Jimmy Lei Ba, Ruslan Salakhutdinov “**Generating Images from Captions with Attention**.” a model that generates images from natural language descriptions. [[pdf]](https://arxiv.org/pdf/1511.02793.pdf)

Jimmy Ba, Geoffrey Hinton, Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu “**Using Fast Weights to Attend to the Recent Past**.” [[pdf]](https://arxiv.org/pdf/1610.06258.pdf)

Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio “**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**.” attention based model that automatically learns to describe the content of images [[pdf]](https://arxiv.org/pdf/1502.03044.pdf)

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun “**Deep Residual Learning for Image Recognition**.” Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously [[pdf]](https://arxiv.org/pdf/1512.03385.pdf)

### 4.4 Autoencoders

“**Introduction to autoencoders**” Explains the first part of Chapter 14 Autoencoder [[Link]](https://www.jeremyjordan.me/autoencoders/)

“**Sparse, Stacked and Variational Autoencoder**”[[Link]](https://medium.com/@venkatakrishna.jonnalagadda/sparse-stacked-and-variational-autoencoder-efe5bfe73b64)

“**Neural Networks, Manifolds, and Topology**” This article will help in understanding the manifold concept in the same chapter [[Link]](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

“**Deep Learning Tutorial - Sparse Autoencoder**” Covers mathematics of Autoencoders for better understanding of the error and penalty functions [[Link]](https://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)

#### 4.4.1 KL Divergence

“**Demystifying KL Divergence**” [[Link]](https://towardsdatascience.com/demystifying-kl-divergence-7ebe4317ee68)

“**Making sense of the Kullback–Leibler (KL) Divergence**”[[Link]](https://medium.com/@cotra.marko/making-sense-of-the-kullback-leibler-kl-divergence-b0d57ee10e0a)

“**Neural Networks, Manifolds, and Topology**” This article will help in understanding the manifold concept in the same chapter [[Link]](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

“**KL Divergence for Machine Learning**” Advanced level articles highlight KL divergence in Deep learning and Reinforcement learning. It alsp educates about forward and backward KL divergence [[Link]](https://dibyaghosh.com/blog/probability/kldivergence.html)

# 5 General ML/DL/RL concepts

“**An introduction to entropy, cross entropy and KL divergence in machine learning**.” [[Link]](https://adventuresinmachinelearning.com/cross-entropy-kl-divergence/)
