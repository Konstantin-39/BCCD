import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Параметры данных
data_folder = 'output'  # папка с файлами данных
keep_difficult = True  # использовать объекты, считающиеся сложными для обнаружения?

# Параметры модели
n_classes = len(label_map)  # количество различных типов объектов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # устройство для вычислений (GPU или CPU)

# Параметры обучения
checkpoint = None  # путь к контрольной точке модели, None, если отсутствует
batch_size = 8  # размер батча
iterations = 120000  # количество итераций для обучения
workers = 4  # количество воркеров для загрузки данных в DataLoader
print_freq = 200  # печать статуса обучения каждые __ батчей
lr = 1e-3  # начальная скорость обучения
decay_lr_at = [80000, 100000]  # уменьшить скорость обучения после этих итераций
decay_lr_to = 0.1  # уменьшить скорость обучения до этой доли текущей скорости
momentum = 0.9  # момент для оптимизатора
weight_decay = 5e-4  # вес для L2-регуляризации
grad_clip = None  # обрезка градиентов, если они взрываются (может происходить при больших размерах батчей)

cudnn.benchmark = True  # ускоряет обучение на GPU, если размеры входных данных постоянны


def main():
    """
    Основная функция для обучения модели.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Инициализация модели или загрузка контрольной точки
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)  # создание модели SSD300
        # Инициализация оптимизатора с удвоенной скоростью обучения для bias-параметров
        biases = []
        not_biases = []
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        # Загрузка контрольной точки
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f'\nЗагружена контрольная точка с эпохи {start_epoch}.\n')
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Перемещение модели и функции потерь на устройство (GPU/CPU)
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Загрузка данных
    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # использование collate_fn для обработки данных

    # Вычисление общего количества эпох и эпох для уменьшения скорости обучения
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # Обучение по эпохам
    for epoch in range(start_epoch, epochs):
        # Уменьшение скорости обучения на определенных эпохах
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # Обучение на одной эпохе
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

        # Сохранение контрольной точки
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Обучение модели на одной эпохе.

    :param train_loader: DataLoader для обучающих данных
    :param model: модель
    :param criterion: функция потерь MultiBoxLoss
    :param optimizer: оптимизатор
    :param epoch: номер текущей эпохи
    """
    model.train()  # перевод модели в режим обучения (включает dropout, если он есть)

    batch_time = AverageMeter()  # время для forward и backward pass
    data_time = AverageMeter()  # время загрузки данных
    losses = AverageMeter()  # значение потерь

    start = time.time()

    # Итерация по батчам
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Перемещение данных на устройство (GPU/CPU)
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward pass
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Вычисление потерь
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # скалярное значение потерь

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Обрезка градиентов, если необходимо
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Обновление параметров модели
        optimizer.step()

        # Обновление метрик
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Печать статуса обучения
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

    # # Освобождение памяти
    # del predicted_locs, predicted_scores, images, boxes, labels


if __name__ == '__main__':
    main()