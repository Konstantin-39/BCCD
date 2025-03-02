import os
import json
import xml.etree.ElementTree as ET

from utils import create_data_lists

if __name__ == '__main__':
    
    create_data_lists(
        annotations_dir='BCCD/Annotations',  # Путь к папке с аннотациями
        imagesets_dir='BCCD/ImageSets/Main',  # Путь к папке с файлами train.txt, test.txt и т.д.
        output_folder='output'  # Папка для сохранения JSON-файлов
    )

