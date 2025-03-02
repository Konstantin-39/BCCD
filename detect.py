from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка модели
checkpoint_path = 'checkpoint_ssd300.pth.tar'
try:
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f'\nЗагружена контрольная точка из эпохи {start_epoch}.\n')
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
except FileNotFoundError:
    logger.error(f"Файл контрольной точки {checkpoint_path} не найден.")
    raise
except Exception as e:
    logger.error(f"Ошибка при загрузке контрольной точки: {e}")
    raise

# Преобразования изображений
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image: Image.Image, min_score: float, max_overlap: float, top_k: int, suppress: List[str] = None) -> Image.Image:
    """
    Обнаруживает объекты на изображении с помощью обученной модели SSD300 и визуализирует результаты.

    :param original_image: изображение, объект PIL.Image
    :param min_score: минимальный порог уверенности для обнаружения объекта
    :param max_overlap: максимальное перекрытие двух bounding boxes, при котором один из них подавляется с помощью Non-Maximum Suppression (NMS)
    :param top_k: если обнаружено много объектов, оставить только top_k с наибольшей уверенностью
    :param suppress: список классов, которые нужно игнорировать
    :return: аннотированное изображение, объект PIL.Image
    """
    try:
        # Преобразование изображения
        image = normalize(to_tensor(resize(original_image)))

        # Перемещение изображения на устройство (GPU/CPU)
        image = image.to(device)

        # Прямой проход через модель
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Обнаружение объектов
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                                 max_overlap=max_overlap, top_k=top_k)

        # Перемещение bounding boxes на CPU
        det_boxes = det_boxes[0].to('cpu')

        # Преобразование bounding boxes к исходным размерам изображения
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Декодирование меток классов
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # Если объекты не найдены, возвращаем исходное изображение
        if det_labels == ['background']:
            logger.info("Объекты не обнаружены.")
            return original_image

        # Аннотирование изображения
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        # Игнорирование указанных классов
        for i in range(det_boxes.size(0)):
            if suppress is not None and det_labels[i] in suppress:
                continue

            # Отрисовка bounding box
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[det_labels[i]])

            # Отрисовка текста с меткой класса
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white', font=font)

        del draw
        return annotated_image

    except Exception as e:
        logger.error(f"Ошибка при обнаружении объектов: {e}")
        raise


if __name__ == '__main__':
    # Пример использования
    img_path = 'D:\Progect\BCCD\BCCD\JPEGImages\BloodImage_00000.jpg'
    try:
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        annotated_image = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
        annotated_image.show()
    except FileNotFoundError:
        logger.error(f"Файл изображения {img_path} не найден.")
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {e}")