from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Хорошее форматирование при выводе AP для каждого класса и mAP
pp = PrettyPrinter()

# Параметры
data_folder = 'output' 
keep_difficult = True  # сложные объекты должны всегда учитываться при расчете mAP, так как они существуют!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = './checkpoint_ssd300.pth.tar'

# Загрузка модели для оценки
try:
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model = model.to(device)
    logger.info("Модель успешно загружена.")
except FileNotFoundError:
    logger.error(f"Файл контрольной точки {checkpoint_path} не найден.")
    raise
except Exception as e:
    logger.error(f"Ошибка при загрузке контрольной точки: {e}")
    raise

# Переключение модели в режим оценки
model.eval()

# Загрузка тестовых данных
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader: torch.utils.data.DataLoader, model: torch.nn.Module) -> None:
    """
    Оценка модели на тестовых данных.

    :param test_loader: DataLoader для тестовых данных
    :param model: модель для оценки
    """
    # Убедимся, что модель в режиме оценки
    model.eval()

    # Списки для хранения обнаруженных и истинных bounding boxes, меток и оценок
    det_boxes = []
    det_labels = []
    det_scores = []
    true_boxes = []
    true_labels = []
    true_difficulties = []  # необходимо знать, какие объекты являются 'сложными'

    with torch.no_grad():
        # Обработка батчей
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Прямой проход через модель
            predicted_locs, predicted_scores = model(images)

            # Обнаружение объектов в выходе SSD
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Оценка ДОЛЖНА быть с min_score=0.01, max_overlap=0.45, top_k=200 для корректного сравнения с результатами из статьи и других репозиториев

            # Сохранение результатов этого батча для расчета mAP
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Расчет mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Вывод AP для каждого класса
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)