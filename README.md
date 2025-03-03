---

# Цель

---

**Построить модель, способную обнаруживать и локализовать определенные объекты на изображениях.**


Мы будем внедрять [Single Shot Multibox Detector (SSD)](https://arxiv.org/abs/1512.02325), решать задачу детекции.

---

# Выполнение

---

В разделах ниже кратко описывается реализация.

### Набор данных

Мы будем использовать набор данных BCCD. BCCD Dataset — это небольшой набор данных для обнаружения клеток крови.

```python
{'RBC', 'WBC', 'Platelets'}
```

Каждое изображение может содержать одино или несколько изображений.

У нас есть три вида клеток крови:

- Эритроцит (красная кровяная клетка)

- WBC (белые кровяные клетки)
  
- Тромбоциты

-  воспринимаемая трудность обнаружения (либо `0`, что означает _не трудно_ , либо `1`, что означает _трудно_ )

* Структура `BCCD_dataset`
  
```
  ├── BCCD
  │   ├── Annotations
  │   │       └── BloodImage_00XYZ.xml (364 items)
  │   ├── ImageSets       # Contain four Main/*.txt which split the dataset
  │   └── JPEGImages
  │       └── BloodImage_00XYZ.jpg (364 items)
  ├── dataset
  │   └── mxnet           
  ├── output
  │   ├── label_map.json        
  │   └── TEST_images.json
  │   └── TEST_objects.json
  │   └── TRAIN_images.json
  │   └── TRAIN_objects.json
  ├── create_data_lists.py        
  ├── datasets.py
  ├── detect.py
  ├── eval.py
  ├── export.py
  ├── model.py
  ├── README.md
  ├── test.csv
  ├── train.py
  ├── utils.py
  ```

#### Скачать

Вам необходимо загрузить следующие наборы данных ВССВ датасет –

- [ВССВ датасет](https://github.com/Shenggan/BCCD_Dataset/tree/master/BCCD) (460MB)

### Входные данные для модели

We will need three inputs.

#### Изображения

Поскольку мы используем вариант SSD300, изображения должны быть размером в `300, 300` пикселях и в формате RGB.

Мы используем базу VGG-16, предварительно обученную на ImageNet, которая уже доступна в модуле PyTorch `torchvision`. [На этой странице](https://pytorch.org/docs/master/torchvision/models.html) подробно описывается предварительная обработка или преобразование, которые нам нужно будет выполнить для использования этой модели — значения пикселей должны быть в диапазоне [0,1], а затем мы должны нормализовать изображение по среднему значению и стандартному отклонению каналов RGB изображений ImageNet.

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Кроме того, PyTorch следует соглашению NCHW, которое означает, что размер каналов (C) должен предшествовать размеру размера.

Следовательно, **изображения, подаваемые в модель, должны представлять собойa `Float` tensor размер `N, 3, 300, 300`**, и должны быть нормализованы по указанному выше среднему значению и стандартному отклонению. `N` - размер пакета.

#### Objects' Bounding Boxes

Для каждого изображения нам необходимо указать ограничивающие рамки присутствующих на нем объектов наземной репрезентативности в дробных граничных координатах `(x_min, y_min, x_max, y_max)`.

Поскольку количество объектов на любом изображении может варьироваться, мы не можем использовать тензор фиксированного размера для хранения ограничивающих рамок для всего пакета `N` изображений.

Таким образом, **ограничивающие рамки истинности, передаваемые в модель, должны представлять собой список длины `N`, где каждый элемент списка является `Float` tensor размером `N_o, 4`**, где `N_o` количество объектов, присутствующих на данном конкретном изображении.

#### Objects' Labels

Для каждого изображения нам необходимо указать метки присутствующих на нем объектов наземной репрезентативности.

Каждая метка должна быть закодирована как целое число от `1` to `3` представляющее три различных типов объектов. Кроме того, мы добавим _background_ (фоновый класс) class с индексом `0`, который указывает на отсутствие объекта в ограничивающем прямоугольнике. (Но, естественно, эта метка фактически не будет использоваться ни для одного из объектов-истин в наборе данных.)

Опять же, поскольку количество объектов на любом изображении может варьироваться, мы не можем использовать тензор фиксированного размера для хранения меток для всего пакета `N` изображений.

Таким образом, **метки истинности, передаваемые в модель, должны представлять собой список длины `N`, где каждый элемент списка является `Long` tensor размером `N_o`**, где `N_o` количество объектов, присутствующих на данном конкретном изображении.

### Data pipeline

As you know, our data is divided into _training_ and _test_ splits.

#### Parse raw data

See `create_data_lists()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This parses the data downloaded and saves the following files –

- A **JSON file for each split with a list of the absolute filepaths of `I` images**, where `I` is the total number of images in the split.

- A **JSON file for each split with a list of `I` dictionaries containing ground truth objects, i.e. bounding boxes in absolute boundary coordinates, their encoded labels, and perceived detection difficulties**. The `i`th dictionary in this list will contain the objects present in the `i`th image in the previous JSON file.

- A **JSON file which contains the `label_map`**, the label-to-index dictionary with which the labels are encoded in the previous JSON file. This dictionary is also available in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) and directly importable.

#### PyTorch Dataset

See `PascalVOCDataset` in [`datasets.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/datasets.py).

This is a subclass of PyTorch [`Dataset`](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset), used to **define our training and test datasets.** It needs a `__len__` method defined, which returns the size of the dataset, and a `__getitem__` method which returns the `i`th image, bounding boxes of the objects in this image, and labels for the objects in this image, using the JSON files we saved earlier.

You will notice that it also returns the perceived detection difficulties of each of these objects, but these are not actually used in training the model. They are required only in the [Evaluation](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#evaluation) stage for computing the Mean Average Precision (mAP) metric. We also have the option of filtering out _difficult_ objects entirely from our data to speed up training at the cost of some accuracy.

Additionally, inside this class, **each image and the objects in them are subject to a slew of transformations** as described in the paper and outlined below.

#### Data Transforms

See `transform()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py).

This function applies the following transformations to the images and the objects in them –

- Randomly **adjust brightness, contrast, saturation, and hue**, each with a 50% chance and in random order.

- With a 50% chance, **perform a _zoom out_ operation** on the image. This helps with learning to detect small objects. The zoomed out image must be between `1` and `4` times as large as the original. The surrounding space could be filled with the mean of the ImageNet data.

- Randomly crop image, i.e. **perform a _zoom in_ operation.** This helps with learning to detect large or partial objects. Some objects may even be cut out entirely. Crop dimensions are to be between `0.3` and `1` times the original dimensions. The aspect ratio is to be between `0.5` and `2`. Each crop is made such that there is at least one bounding box remaining that has a Jaccard overlap of either `0`, `0.1`, `0.3`, `0.5`, `0.7`, or `0.9`, randomly chosen, with the cropped image. In addition, any bounding boxes remaining whose centers are no longer in the image as a result of the crop are discarded. There is also a chance that the image is not cropped at all.

- With a 50% chance, **horizontally flip** the image.

- **Resize** the image to `300, 300` pixels. This is a requirement of the SSD300.

- Convert all boxes from **absolute to fractional boundary coordinates.** At all stages in our model, all boundary and center-size coordinates will be in their fractional forms.

- **Normalize** the image with the mean and standard deviation of the ImageNet data that was used to pretrain our VGG base.

As mentioned in the paper, these transformations play a crucial role in obtaining the stated results.

#### PyTorch DataLoader

The `Dataset` described above, `PascalVOCDataset`, will be used by a PyTorch [`DataLoader`](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) in `train.py` to **create and feed batches of data to the model** for training or evaluation.

Since the number of objects vary across different images, their bounding boxes, labels, and difficulties cannot simply be stacked together in the batch. There would be no way of knowing which objects belong to which image.

Instead, we need to **pass a collating function to the `collate_fn` argument**, which instructs the `DataLoader` about how it should combine these varying size tensors. The simplest option would be to use Python lists.

### Base Convolutions

See `VGGBase` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply base convolutions.**

The layers are initialized with parameters from a pretrained VGG-16 with the `load_pretrained_layers()` method.

We're especially interested in the lower-level feature maps that result from `conv4_3` and `conv7`, which we return for use in subsequent stages.

### Auxiliary Convolutions

See `AuxiliaryConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply auxiliary convolutions.**

Use a [uniform Xavier initialization](https://pytorch.org/docs/stable/nn.html#torch.nn.init.xavier_uniform_) for the parameters of these layers.

We're especially interested in the higher-level feature maps that result from `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, which we return for use in subsequent stages.

### Prediction Convolutions

See `PredictionConvolutions` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, we **create and apply localization and class prediction convolutions** to the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`.

These layers are initialized in a manner similar to the auxiliary convolutions.

We also **reshape the resulting prediction maps and stack them** as discussed. Note that reshaping in PyTorch is only possible if the original tensor is stored in a [contiguous](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.contiguous) chunk of memory.

As expected, the stacked localization and class predictions will be of dimensions `8732, 4` and `8732, 21` respectively.

### Putting it all together

See `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Here, the **base, auxiliary, and prediction convolutions are combined** to form the SSD.

There is a small detail here – the lowest level features, i.e. those from `conv4_3`, are expected to be on a significantly different numerical scale compared to its higher-level counterparts. Therefore, the authors recommend L2-normalizing and then rescaling _each_ of its channels by a learnable value.

### Priors

See `create_prior_boxes()` under `SSD300` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

This function **creates the priors in center-size coordinates** as defined for the feature maps from `conv4_3`, `conv7`, `conv8_2`, `conv9_2`, `conv10_2` and `conv11_2`, _in that order_. Furthermore, for each feature map, we create the priors at each tile by traversing it row-wise.

This ordering of the 8732 priors thus obtained is very important because it needs to match the order of the stacked predictions.

### Multibox Loss

See `MultiBoxLoss` in [`model.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py).

Two empty tensors are created to store localization and class prediction targets, i.e. _ground truths_, for the 8732 predicted boxes in each image.

We **find the ground truth object with the maximum Jaccard overlap for each prior**, which is stored in `object_for_each_prior`.

We want to avoid the rare situation where not all of the ground truth objects have been matched. Therefore, we also **find the prior with the maximum overlap for each ground truth object**, stored in `prior_for_each_object`. We explicitly add these matches to `object_for_each_prior` and artificially set their overlaps to a value above the threshold so they are not eliminated.

Based on the matches in `object_for_each prior`, we set the corresponding labels, i.e. **targets for class prediction**, to each of the 8732 priors. For those priors that don't overlap significantly with their matched objects, the label is set to _background_.

Also, we encode the coordinates of the 8732 matched objects in `object_for_each prior` in offset form `(g_c_x, g_c_y, g_w, g_h)` with respect to these priors, to form the **targets for localization**. Not all of these 8732 localization targets are meaningful. As we discussed earlier, only the predictions arising from the non-background priors will be regressed to their targets.

The **localization loss** is the [Smooth L1 loss](https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss) over the positive matches.

Perform Hard Negative Mining – rank class predictions matched to _background_, i.e. negative matches, by their individual [Cross Entropy losses](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss). The **confidence loss** is the Cross Entropy loss over the positive matches and the hardest negative matches. Nevertheless, it is averaged only by the number of positive matches.

The **Multibox Loss is the aggregate of these two losses**, combined in the ratio `α`. In our case, they are simply being added because `α = 1`.

# Training

Before you begin, make sure to save the required data files for training and evaluation. To do this, run the contents of [`create_data_lists.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/create_data_lists.py) after pointing it to the `VOC2007` and `VOC2012` folders in your [downloaded data](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#download).

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you need to.

To **train your model from scratch**, run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

### Remarks

In the paper, they recommend using **Stochastic Gradient Descent** in batches of `32` images, with an initial learning rate of `1e−3`, momentum of `0.9`, and `5e-4` weight decay.

I ended up using a batch size of `8` images for increased stability. If you find that your gradients are exploding, you could reduce the batch size, like I did, or clip gradients.

The authors also doubled the learning rate for bias parameters. As you can see in the code, this is easy do in PyTorch, by passing [separate groups of parameters](https://pytorch.org/docs/stable/optim.html#per-parameter-options) to the `params` argument of its [SGD optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD).

The paper recommends training for 80000 iterations at the initial learning rate. Then, it is decayed by 90% (i.e. to a tenth) for an additional 20000 iterations, _twice_. With the paper's batch size of `32`, this means that the learning rate is decayed by 90% once after the 154th epoch and once more after the 193th epoch, and training is stopped after 232 epochs. I followed this schedule.

On a TitanX (Pascal), each epoch of training required about 6 minutes.

I should note here that two unintended differences from the paper were brought to my attention by readers of this tutorial:

- My priors that overshoot the edges of the image are not being clipped, as pointed out in issue [#94](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94) by _@AakiraOtok_. This does not appear to have a negative effect on performance, however, as discussed in that issue and also verified in issue [#95](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/95) by the same reader. It is even possible that there is a slight improvement in performance, but this may be too small to be conclusive.

- I mistakenly used L1 loss instead of *smooth* L1 loss as the localization loss, as pointed out in issue [#60](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/60) by _jonathan016_. This also appears to have no negative effect on performance as pointed out in that issue, but _smooth_ L1 loss may offer better training stability with larger batch sizes as mentioned in [this comment](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/issues/94#issuecomment-1590217018). 

### Model checkpoint

You can download this pretrained model [here](https://drive.google.com/open?id=1bvJfF6r_zYl2xZEpYXxgb7jLQHFZ01Qe).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load) for evaluation or inference – see below.

# Evaluation

See [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/eval.py).

The data-loading and checkpoint parameters for evaluating the model are at the beginning of the file, so you can easily check or modify them should you wish to.

To begin evaluation, simply run the `evaluate()` function with the data-loader and model checkpoint. **Raw predictions for each image in the test set are obtained and parsed** with the checkpoint's `detect_objects()` method, which implements [this process](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#processing-predictions). Evaluation has to be done at a `min_score` of `0.01`, an NMS `max_overlap` of `0.45`, and `top_k` of `200` to allow fair comparision of results with the paper and other implementations.

**Parsed predictions are evaluated against the ground truth objects.** The evaluation metric is the _Mean Average Precision (mAP)_. If you're not familiar with this metric, [here's a great explanation](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

We will use `calculate_mAP()` in [`utils.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py) for this purpose. As is the norm, we will ignore _difficult_ detections in the mAP calculation. But nevertheless, it is important to include them from the evaluation dataset because if the model does detect an object that is considered to be _difficult_, it must not be counted as a false positive.

The model scores **77.2 mAP**, same as the result reported in the paper.

Class-wise average precisions (not scaled to 100) are listed below.

| Class | Average Precision |
| :-----: | :------: |
| _aeroplane_ | 0.7887580990791321 |
| _bicycle_ | 0.8351995348930359 |
| _bird_ | 0.7623348236083984 |
| _boat_ | 0.7218425273895264 |
| _bottle_ | 0.45978495478630066 |
| _bus_ | 0.8705356121063232 |
| _car_ | 0.8655831217765808 |
| _cat_ | 0.8828985095024109 |
| _chair_ | 0.5917483568191528 |
| _cow_ | 0.8255912661552429 |
| _diningtable_ | 0.756867527961731 |
| _dog_ | 0.856262743473053 |
| _horse_ | 0.8778411149978638 |
| _motorbike_ | 0.8316892385482788 |
| _person_ | 0.7884440422058105 |
| _pottedplant_ | 0.5071538090705872 |
| _sheep_ | 0.7936667799949646 |
| _sofa_ | 0.7998116612434387 |
| _train_ | 0.8655905723571777 |
| _tvmonitor_ | 0.7492395043373108 |

You can see that some objects, like bottles and potted plants, are considerably harder to detect than others.

# Inference

See [`detect.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/detect.py).

Point to the model you want to use for inference with the `checkpoint` parameter at the beginning of the code.

Then, you can use the `detect()` function to identify and visualize objects in an RGB image.

```python
img_path = '/path/to/ima.ge'
original_image = PIL.Image.open(img_path, mode='r')
original_image = original_image.convert('RGB')

detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
```

This function first **preprocesses the image by resizing and normalizing its RGB channels** as required by the model. It then **obtains raw predictions from the model, which are parsed** by the `detect_objects()` method in the model. The parsed results are converted from fractional to absolute boundary coordinates, their labels are decoded with the `label_map`, and they are **visualized on the image**.

There are no one-size-fits-all values for `min_score`, `max_overlap`, and `top_k`. You may need to experiment a little to find what works best for your target data.

### Some more examples

---

<p align="center">
<img src="./img/000029.jpg">
</p>

---

<p align="center">
<img src="./img/000045.jpg">
</p>

---

<p align="center">
<img src="./img/000062.jpg">
</p>

---

<p align="center">
<img src="./img/000075.jpg">
</p>

---

<p align="center">
<img src="./img/000085.jpg">
</p>

---

<p align="center">
<img src="./img/000092.jpg">
</p>

---

<p align="center">
<img src="./img/000100.jpg">
</p>

---

<p align="center">
<img src="./img/000124.jpg">
</p>

---

<p align="center">
<img src="./img/000127.jpg">
</p>

---

<p align="center">
<img src="./img/000128.jpg">
</p>

---

<p align="center">
<img src="./img/000145.jpg">
</p>

---

# FAQs

__I noticed that priors often overshoot the `3, 3` kernel employed in the prediction convolutions. How can the kernel detect a bound (of an object) outside it?__

Don't confuse the kernel and its _receptive field_, which is the area of the original image that is represented in the kernel's field-of-view.

For example, on the `38, 38` feature map from `conv4_3`, a `3, 3` kernel covers an area of `0.08, 0.08` in fractional coordinates. The priors are `0.1, 0.1`, `0.14, 0.07`, `0.07, 0.14`, and `0.14, 0.14`.

But its receptive field, which [you can calculate](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807), is a whopping `0.36, 0.36`! Therefore, all priors (and objects contained therein) are present well inside it.

Keep in mind that the receptive field grows with every successive convolution. For `conv_7` and the higher-level feature maps, a `3, 3` kernel's receptive field will cover the _entire_ `300, 300` image. But, as always, the pixels in the original image that are closer to the center of the kernel have greater representation, so it is still _local_ in a sense.

---

__While training, why can't we match predicted boxes directly to their ground truths?__

We cannot directly check for overlap or coincidence between predicted boxes and ground truth objects to match them because predicted boxes are not to be considered reliable, _especially_ during the training process. This is the very reason we are trying to evaluate them in the first place!

And this is why priors are especially useful. We can match a predicted box to a ground truth box by means of the prior it is supposed to be approximating. It no longer matters how correct or wildly wrong the prediction is.

---

__Why do we even have a _background_ class if we're only checking which _non-background_ classes meet the threshold?__

When there is no object in the approximate field of the prior, a high score for _background_ will dilute the scores of the other classes such that they will not meet the detection threshold.

---

__Why not simply choose the class with the highest score instead of using a threshold?__

I think that's a valid strategy. After all, we implicitly conditioned the model to choose _one_ class when we trained it with the Cross Entropy loss. But you will find that you won't achieve the same performance as you would with a threshold.

I suspect this is because object detection is open-ended enough that there's room for doubt in the trained model as to what's really in the field of the prior. For example, the score for _background_ may be high if there is an appreciable amount of backdrop visible in an object's bounding box. There may even be multiple objects present in the same approximate region. A simple threshold will yield all possibilities for our consideration, and it just works better.

Redundant detections aren't really a problem since we're NMS-ing the hell out of 'em.


---

__Sorry, but I gotta ask... _[what's in the boooox?!](https://cnet4.cbsistatic.com/img/cLD5YVGT9pFqx61TuMtcSBtDPyY=/570x0/2017/01/14/6d8103f7-a52d-46de-98d0-56d0e9d79804/se7en.png)___

Ha.
